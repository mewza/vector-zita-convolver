// ----------------------------------------------------------------------------
//
//  Copyright (C) 2006-2018 Fons Adriaensen <fons@linuxaudio.org>
//  Vectorization (C) 2024 Dmitry Boldyrev 
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even thsure implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// ----------------------------------------------------------------------------


#pragma once

#include <pthread.h>
#include <stdint.h>
#include <simd/simd.h>
#include "const1.h"
#include "pffft.h"
#include "pffft_double.h"
#include "avfft.h"

#define ENABLE_VECTOR_MODE  1

// replaces serial version of PFFFT with serial
// non-vector version of AVFFT

// Enable EQ application (check clv_process)

static constexpr double DEFAULT_MAC_COST = 1.0;
static constexpr double DEFAULT_FFT_COST = 5.0;

enum {
    MAXINP   = 8,
    MAXOUT   = 8,
    MAXLEV   = 8,
    MINPART  = 64,
    MAXPART  = 8192,
    MAXDIVIS = 16,
    MINQUANT = 16,
    MAXQUANT = 8192
};

#ifndef D_FFT_MODE
#define D_FFT_MODE

enum FFT_MODE {
    USE_AVFFT,
    USE_PFFFT
};

#endif


class Converror
{
public:
    enum {
        BAD_STATE = -1,
        BAD_PARAM = -2,
        MEM_ALLOC = -3
    };
    
    Converror (int error) : _error (error) {}
    
private:
    
    int _error;
};

extern float gVolume;

typedef zfloat FV4 __attribute__ ((vector_size(sizeof(zfloat)*4)));

template <typename T, FFT_MODE mode, bool OVERRIDE_PFFFT> class Convlevel;
template <typename T, FFT_MODE mode, bool OVERRIDE_PFFFT> class Convproc;

static int zita_convolver_major_version (void) {
    return 4;
}

static int zita_convolver_minor_version (void) {
    return 1;
}

#ifdef ZCSEMA_IS_IMPLEMENTED
#undef ZCSEMA_IS_IMPLEMENTED
#endif

#if defined(__linux__)  || defined(__GNU__) || defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
#include <semaphore.h>

class ZCsema
{
public:
    
    ZCsema (void) { init (0, 0); }
    ~ZCsema (void) { sem_destroy (&_sema); }
    
    ZCsema (const ZCsema&); // disabled
    ZCsema& operator= (const ZCsema&); // disabled
    
    int init (int s, int v) { return sem_init (&_sema, s, v); }
    int post (void) { return sem_post (&_sema); }
    int wait (void) { return sem_wait (&_sema); }
    int trywait (void) { return sem_trywait (&_sema); }
    
private:
    
    sem_t  _sema;
};

#else

#ifdef __APPLE__

// NOTE:  ***** I DO NOT REPEAT NOT PROVIDE SUPPORT FOR OSX *****
//
// The following code partially emulates the POSIX sem_t for which
// OSX has only a crippled implementation. It may or may not compile,
// and if it compiles it may or may not work correctly. Blame APPLE
// for not following POSIX standards.

class ZCsema
{
public:
    
    ZCsema (void) : _count (0)
    {
        init (0, 0);
    }
    
    ~ZCsema (void)
    {
        pthread_mutex_destroy (&_mutex);
        pthread_cond_destroy (&_cond);
    }
    
    ZCsema (const ZCsema&); // disabled
    ZCsema& operator= (const ZCsema&); // disabled
    
    int init (int s, int v)
    {
        _count = v;
        return pthread_mutex_init (&_mutex, 0) || pthread_cond_init (&_cond, 0);
    }
    
    int post (void)
    {
        pthread_mutex_lock (&_mutex);
        _count++;
        if (_count == 1) pthread_cond_signal (&_cond);
        pthread_mutex_unlock (&_mutex);
        return 0;
    }
    
    int wait (void)
    {
        pthread_mutex_lock (&_mutex);
        while (_count < 1) pthread_cond_wait (&_cond, &_mutex);
          _count--;
        pthread_mutex_unlock (&_mutex);
        return 0;
    }
    
    int trywait (void)
    {
        if (pthread_mutex_trylock (&_mutex)) return -1;
        if (_count < 1)
        {
            pthread_mutex_unlock (&_mutex);
            return -1;
        }
        _count--;
        pthread_mutex_unlock (&_mutex);
        return 0;
    }
    
private:
    
    int              _count;
    pthread_mutex_t  _mutex;
    pthread_cond_t   _cond;
};

#define ZCSEMA_IS_IMPLEMENTED
#endif

#ifndef ZCSEMA_IS_IMPLEMENTED
#error "The ZCsema class is not implemented."
#endif

#endif

template <typename T, FFT_MODE mode, bool OVERRIDE_PFFFT>
class Convlevel
{
    friend class Convproc<T, mode, OVERRIDE_PFFFT>;
    static constexpr bool doUseVector = ((std::is_same_v<T, float> || std::is_same_v<T, double>) && ENABLE_VECTOR_MODE == 1);

    class Inpnode
    {
        friend class Convlevel<T, mode, OVERRIDE_PFFFT>;
     public:
        
        Inpnode(uint16_t inp) :
        _next (0),
        _ffta (0),
        _npar (0),
        _inp (inp),
        _islink (false)
        {}
        
        ~Inpnode (void)
        {
            if (!_ffta) return;
            for (uint16_t i = 0; i < _npar; i++) {
                if (!_islink && _ffta[i])
                    callocT_free (_ffta [i]);
            }
            if (!_islink)
                callocT_free(_ffta);
            _ffta = 0;
            _npar = 0;
            _islink = false;
        }
        
        __inline void link_ffta (Inpnode *root)
        {
            _islink = true;
            _npar = root->_npar;
            _ffta = root->_ffta;
        }
        __inline void alloc_ffta_pffft (uint16_t npar, int32_t size)
        {
            _npar = npar;
            _islink = false;
            
            _ffta = (void**) callocT<void*>(_npar); THROW_MEM( _ffta == NULL );
            for (int i = 0; i < _npar; i++) {
                _ffta[i] = callocT<cmplxT<zfloat>>( size + 1 ); THROW_MEM( _ffta [i] == NULL );
            }
        }
        __inline void alloc_ffta_avfft (uint16_t npar, int32_t size)
        {
            _npar = npar;
            _islink = false;
            
            _ffta = (void**) callocT<void*>(_npar); THROW_MEM( _ffta == NULL );
            for (int i = 0; i < _npar; i++) {
                _ffta[i] = callocT<cmplxT<zfloat8>>( size + 1 ); THROW_MEM( _ffta [i] == NULL );
            }
        }
    protected:
        Inpnode        *_next;
        void          **_ffta;
        int             _npar;
        int             _inp;
        bool            _islink;
    };

    class Macnode
    {
        friend class Convlevel<T, mode, OVERRIDE_PFFFT>;
    public:
        
        Macnode (Inpnode *inpn):
            _next (0),
            _inpn (inpn),
            _link (0),
            _fftb (0),
            _npar (0),
            _chn (0),
            _inp (0),
            _out (0),
            _inpmask (0),
            _outmask (0),
            _inpneg (0.0),
            _outneg (0.0),
            _islink (false)
        {}

        ~Macnode (void)
        {
            free_fftb();
        }

        void alloc_fftb (int npar,int chn, int inp, int out)
        {
            _npar = npar;
            _chn = chn;
            _inp = inp;
            _out = out;
            _islink = false;
            _fftb = (void**)callocT<void*>( _npar );
            for (uint16_t i = 0; i < _npar; i++)
                _fftb[i] = NULL;
        }
        
        void link_fftb (int chn, int inp, int out, Macnode *link)
        {
            _npar = link->_npar;
            _fftb = link->_fftb;
            _outmask = link->_outmask;
            _inpmask = link->_inpmask;
            _inpneg = link->_inpneg;
            _outneg = link->_outneg;
            _islink = true;
            _chn = chn;
            _inp = inp;
            _out = out;
        }
        
        void free_fftb (void)
        {
            if (!_islink && _fftb) {
                for (int i = 0; i < _npar; i++)
                    if (_fftb [i]) callocT_free( _fftb[i] );
                callocT_free( _fftb );
            }
            _fftb = 0;
            _npar = 0;
            _inp = 0;
            _out = 0;
            _islink = false;
        }

        Macnode          *_next, *_link;
        Inpnode          *_inpn;
        void            **_fftb;
        int               _npar;
        int               _chn;
        int               _out;
        int               _inp;
        int8v             _inpmask;
        int8v             _outmask;
        zfloat8           _inpneg;
        zfloat8           _outneg;
        bool              _islink;
        
    };

    class Outnode
    {
        friend class Convlevel<T, mode, OVERRIDE_PFFFT>;
    public:
        Outnode (uint16_t out, int32_t size):
            _next (0),
            _list (0),
            _out (out),
            _isroot(true)
        {
            for (int i=0; i<3; i++) {
                 _buff [i] = (T*)callocT<zfloat8>( size );
                _buffo [i] = (T*)callocT<zfloat8>( size );
            }
        }
        Outnode (uint16_t out, Outnode *root):
            _next (0),
            _list (0),
            _out (out),
            _isroot(false)
        {
            for (int i=0; i<3; i++) {
                 _buff [i] = root->_buff [i];
                _buffo [i] = root->_buffo [i];
            }
        }
        ~Outnode (void)
        {
            if (!_isroot) return;
            
            for (int i=0; i<3; i++) {
                if (_buff [i])
                    callocT_free (_buff [i]);
                if (_buffo [i])
                    callocT_free (_buffo [i]);
            }
        }
        
    public:
        Outnode           *_next;
        Macnode           *_list;
        T                 *_buff [3];
        T                 *_buffo [3];
        int               _out;
        bool              _isroot;
    };

    
private:
    
    enum {
        OPT_FFTW_MEASURE = 1,
        OPT_VECTOR_MODE  = 2,
        OPT_LATE_CONTIN  = 4
    };
    
    enum {
        ST_IDLE,
        ST_TERM,
        ST_PROC
    };
    
    Convlevel () :
        _stat (ST_IDLE),
        _npar (0),
        _parsize (0),
        _options (OPT_VECTOR_MODE),
        _pthr (0),
        _inp_list (0),
        _out_list (0),
        _ir_list (0),
        _time_data (0),
        _prep_data (0),
        _plan_r2c (0),
        _plan_c2r (0),
        _freq_data (0),
        _fftCycle (0),
        _cycleTot (0),
        _eq_active (false)
    {
    }

    ~Convlevel (void)
    {
        cleanup ();
        
        if (_plan_r2c) pffft_destroy_setup (_plan_r2c); _plan_r2c = NULL;
        if (_plan_c2r) pffft_destroy_setup (_plan_c2r); _plan_c2r = NULL;
        if (_d_plan_r2c) pffftd_destroy_setup (_d_plan_r2c); _d_plan_r2c = NULL;
        if (_d_plan_c2r) pffftd_destroy_setup (_d_plan_c2r); _d_plan_c2r = NULL;
        
        if (_prep_data) callocT_free (_prep_data); _prep_data = NULL;
        if (_time_data) callocT_free (_time_data); _time_data = NULL;
        if (_freq_data) callocT_free (_freq_data); _freq_data = NULL;
    }

    void configure (int nch, int prio, int offs, int npar, int parsize, int options)
    {
        _prio = prio;
        _offs = offs;
        _npar = npar;
        _parsize = parsize;
        _options = options | OPT_VECTOR_MODE;
        
        _fft = new FFTReal<zfloat>( _parsize * 2 );
        _fft8 = new FFTReal<zfloat8>( _parsize * 2 );
    
        _spec_data = callocT<cmplxT<mssFloat2>>(_parsize); THROW_MEM(!_spec_data);
        
        if constexpr(mode == USE_PFFFT)
        {
            _time_data = (T*)callocT<zfloat> (2 * _parsize); THROW_MEM(!_time_data);
            _freq_data = (cmplxT<T>*)callocT<cmplxT<zfloat>> (_parsize + 1); THROW_MEM(!_freq_data);
            _prep_data = (zfloat*)callocT<zfloat> (2 * _parsize); THROW_MEM(!_prep_data);

            if ZFLOAT_IS_FLOAT {
                _d_plan_r2c = _d_plan_c2r = NULL;
                _plan_r2c = pffft_new_setup( (int)(2 * _parsize), PFFFT_REAL); THROW_MEM(!_plan_r2c);
                _plan_c2r = pffft_new_setup( (int)(2 * _parsize), PFFFT_REAL); THROW_MEM(!_plan_c2r);
            } else {
                _plan_r2c = _plan_c2r = NULL;
                _d_plan_r2c = pffftd_new_setup( (int)(2 * _parsize), PFFFT_REAL); THROW_MEM(!_d_plan_r2c);
                _d_plan_c2r = pffftd_new_setup( (int)(2 * _parsize), PFFFT_REAL); THROW_MEM(!_d_plan_c2r);
            }
        } else {
            _time_data = (T*)callocT<T> (2 * _parsize); THROW_MEM(!_time_data);
            _freq_data = (cmplxT<T>*)callocT<cmplxT<zfloat8>> (_parsize + 1); THROW_MEM(!_freq_data);
            _prep_data = (zfloat*)callocT<zfloat8> (2 * _parsize); THROW_MEM(!_prep_data);

        }
        
        //   hanning_window(_win, _parsize * 4);
    }

    void impdata_write_PFFFT (int chn, int *inp, int *out, int step, float *data, int i0, int i1, int nch, bool create)
    {
        int k, j, j0, j1, n;
        Macnode *M1;
        cmplxT<zfloat> *fftb;
        zfloat norm;
        
        int _inp = inp[chn]-1;
        int _out = out[chn]-1;
        
        _nch = nch;
        n = i1 - i0;
        i0 = _offs - i0;
        i1 = i0 + _npar * _parsize;
        
        cmplxT<zfloat> *freq_data = (cmplxT<zfloat> *)_freq_data;
        if ((i0 >= n) || (i1 <= 0))
            return;
        
        if (create)
        {
            M1 = findmacnode_PFFFT( _inp, _out, true );
            if (!M1 || M1->_link) return;
            if (!M1->_fftb) M1->alloc_fftb( _npar, chn, _inp, _out );
        } else {
            M1 = findmacnode_PFFFT( _inp, _out, false );
            if (!M1 || M1->_link || !M1->_fftb)
                return;
        }
        
        norm = 0.5 / (zfloat)_parsize;
        for (k=0; k<_npar; k++)
        {
            i1 = i0 + _parsize;
            if ((i0 < n) && (i1 > 0))
            {
                fftb = (cmplxT<zfloat>*)M1->_fftb[k];
                if (!fftb && create)
                    M1->_fftb[k] = fftb = callocT<cmplxT<zfloat>> (_parsize + 1);
                if (fftb && data)
                {
                    memset (_prep_data, 0, 2 * _parsize * sizeof (zfloat));
                    
                    j0 = (i0 < 0) ? 0 : i0;
                    j1 = (i1 > n) ? n : i1;
                    
                    for (j = j0; j < j1; j++) {
                        _prep_data[j - i0] = norm * data[j * step];
                    }
                    if constexpr( OVERRIDE_PFFFT )
                        _av.real_fft( (zfloat*)_prep_data, (zfloat*)_freq_data, _parsize * 2, true );
                    else {
                        if ZFLOAT_IS_FLOAT
                            pffft_transform_ordered(_plan_r2c, (float*)_prep_data, (float*)_freq_data, NULL, PFFFT_FORWARD);
                        else
                            pffftd_transform_ordered(_d_plan_r2c, (double*)_prep_data, (double*)_freq_data, NULL, PFFFT_FORWARD);
                    }
                    if constexpr( doUseVector )
                        fftswap ((cmplxT<T> *)_freq_data);

                    for (j = 0; j <= _parsize; j++)
                        fftb [j] += freq_data [j];
                }
            }
            i0 = i1;
        }
    }

    void make_mask(int *data, int chn, int nch, int8v &mask, zfloat8 &neg)
    {
        int i;
        int rows = ceil(nch / 8.);
        int row  = ceil((chn+1) / 8.)-1;
        int ach  = ((ach = nch - ((rows - row) * 8)) < 0) ? 8 : ach;
        
        for (i = 0; i < ach; i++) {
            mask[i] = data[i + row * 8]-1;
            neg[i] = 1;
        }
        for (; i < 8; i++) {
            mask[i + row * 8] = 0;
            neg[i] = 0;
        }
        
       // LOG("chn: %d nch: %d -> ", chn, nch); PRINT_I8(mask);
    }
    
    void impdata_write_AVFFT (int c, int *inp, int *out, int step, float *data, int i0, int i1, int nch, bool create)
    {
        int k, j, j0, j1, n;
        Macnode         *M1;
        cmplxT<zfloat8> *fftb;
        zfloat norm;
        
        int _inp = inp[c]-1;
        int _out = out[c]-1;
        
        // inp:  {   1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 1, 4, 3, 2,  4, 3, 4, 3 };
        // chn:  {   1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,  3, 4,10,11 };
        // out:  {   1, 2, 3, 4, 5, 6, 7, 2, 1, 4, 3, 6, 5, 8,  4, 3, 4, 3 };
        
        _nch = nch;
        n = i1 - i0;
        i0 = _offs - i0;
        i1 = i0 + _npar * _parsize;
        
        if ((i0 >= n) || (i1 <= 0)) return;
        
        int ch = c % 8;
        int rows = ceil((zfloat)nch / 8.);
        int row = ceil((zfloat)(c + 1) / 8.)-1;
        int ach = ((ach = nch - ((rows - row) * 8)) < 0) ? 8 : ach;
        
        cmplxT<zfloat> *freq_data = (cmplxT<zfloat> *)_freq_data;
        auto prep_data = _prep_data;
        
        if (create)
        {
            M1 = findmacnode_AVFFT( c, _inp, _out, true );
            if (!M1 || M1->_link) return;
            if (!ch) {
                M1->alloc_fftb( _npar, c, _inp, _out );
                make_mask(inp, c, nch, M1->_inpmask, M1->_inpneg);
                make_mask(out, c, nch, M1->_outmask, M1->_outneg);
                _mlink = M1;
            } else
                M1->link_fftb( c, _inp, _out, _mlink );
        } else {
            M1 = findmacnode_AVFFT( c, _inp, _out, false );
            if (!M1 || M1->_link || !M1->_fftb) return;
        }
        
      //  LOG("[ %d ] _npar: %d inp: %d out: %d row: %d rows: %d ch: %d ach: %d", chn, _npar, _inp, _out, row, rows, ch+1, ach);
        norm = 0.5 / (zfloat)_parsize;
        for (k = 0; k<_npar; k++)
        {
            i1 = i0 + _parsize;
            if ((i0 < n) && (i1 > 0))
            {
                fftb = (cmplxT<zfloat8>*) M1->_fftb[k];
                if (!fftb && create) 
                {
                    if ((c % 8) == 0) {
                         M1->_fftb[k] = fftb = callocT<cmplxT<zfloat8>>((_parsize + 1) * rows);
                    }
                }
                if (fftb && data)
                {
                    memset (_prep_data, 0, 2 * _parsize * sizeof (zfloat));
                    j0 = (i0 < 0) ? 0 : i0;
                    j1 = (i1 > n) ? n : i1;
                    for (j = j0; j < j1; j++) {
                        prep_data[j - i0] = norm * data [j * step]; // * norm
                    }
                   //if (((chn + 1) % 8) == 0 || chn == nch) {
                    //LOG("=== %d chn: %d nch: %d", k, chn, nch);
                    int offs = row * (_parsize + 1);
                    _av.real_fft( (zfloat*)prep_data, (zfloat*)freq_data, _parsize * 2, true );
                   
                    for (j = 0; j <= _parsize; j++) {
                        fftb[j + offs].re[ch] += freq_data[j].re;
                        fftb[j + offs].im[ch] += freq_data[j].im;
                    }
                    //  PRINT_D8(fftb[0].re);
                  //  }
                    //LOG("%d row: %d chn: %d", k, row, ((chn+1) % 8));
                }
            }
            i0 = i1;
        }
    }
    
    void impdata_clear (int inp, int out)
    {
        Macnode *M1;
        
        if constexpr( mode == USE_PFFFT )
            M1 = findmacnode_PFFFT (inp, out, false);
        else
            M1 = findmacnode_AVFFT (inp, out, NULL, false);
        
        if (!M1 || M1->_link || !M1->_fftb)
            return;
        for (int i = 0; i < _npar; i++)
            if (M1->_fftb [i])
                memset (M1->_fftb [i], 0, (_parsize + 1) * sizeof (cmplxT<zfloat>));
    }

    void impdata_link (int inp1, int out1, int inp2, int out2)
    {
        Macnode *M1;
        Macnode *M2;
        
        M1 = findmacnode_AVFFT (inp1, out1, false);
        if (! M1) return;
        M2 = findmacnode_AVFFT (inp2, out2, true);
        M2->free_fftb();
        M2->_link = M1;
    }

    // USE_AVFFT
    void reset (int inpsize, int outsize, zfloat4 *inpbuff, zfloat8 *outbuff)
    {
        int     i;
        Inpnode *X;
        Outnode *Y;
        
        _inpsize = inpsize;
        _outsize = outsize;
        
        _inpbuff = NULL;
        _outbuff = NULL;
        
        _inpbuff_4 = inpbuff;
        _outbuff_8 = outbuff;

        X = _inp_list;
        if (X)
            for (i = 0; i < _npar; i++)
                memset (X->_ffta [i], 0, (_parsize + 1) * sizeof (cmplxT<zfloat8>));

        Y = _out_list;
        if (Y)
            for (i = 0; i < 3; i++) {
                if (Y->_buff [i])
                    memset (Y->_buff [i], 0, _parsize * sizeof (zfloat8));
                if (Y->_buffo [i])
                    memset (Y->_buffo [i], 0, _parsize * sizeof (zfloat8));
            }
        if (_parsize == _outsize) {
            _outoffs = 0;
            _inpoffs = 0;
        } else {
            _outoffs = _parsize / 2;
            _inpoffs = _inpsize - _outoffs;
        }
        _bits = (int)_parsize / _outsize;
        _wait = 0;
        _ptind = 0;
        _opind = 0;
        _fftCycle = 0;
        _cycleTot = 0;
        _trig.init (0, 0);
        _done.init (0, 0);
    }
    
    // USE_PFFFT
    void reset (int inpsize, int outsize, T **inpbuff, T **outbuff)
    {
        int     i;
        Inpnode *X;
        Outnode *Y;
        
        _inpsize = inpsize;
        _outsize = outsize;
        
        _inpbuff_4 = NULL;
        _outbuff_8 = NULL;
        
        _inpbuff = inpbuff;
        _outbuff = outbuff;

        //X = _inp_list;
        for (X = _inp_list; X; X = X->_next)
            for (i = 0; i < _npar; i++)
                memset (X->_ffta [i], 0, (_parsize + 1) * sizeof (cmplxT<zfloat>));
        
        for (Y = _out_list; Y; Y = Y->_next)
            for (i = 0; i < 3; i++) {
                memset (Y->_buff [i], 0, _parsize * sizeof (zfloat));
                memset (Y->_buffo [i], 0, _parsize * sizeof (zfloat));
            }
       
        if (_parsize == _outsize) {
            _outoffs = 0;
            _inpoffs = 0;
        } else {
            _outoffs = _parsize / 2;
            _inpoffs = _inpsize - _outoffs;
        }
        
        _bits = (int)_parsize / _outsize;
        _wait = 0;
        _ptind = 0;
        _opind = 0;
        _fftCycle = 0;
        _cycleTot = 0;
        _trig.init (0, 0);
        _done.init (0, 0);
    }
    
    void start (int abspri, int policy)
    {
        int                min, max;
        pthread_attr_t     attr;
        struct sched_param parm;
        
        _fftCycle = 0;
        _cycleTot = 0;
        _pthr = 0;
        min = sched_get_priority_min (policy);
        max = sched_get_priority_max (policy);
        abspri += _prio;
        if (abspri > max) abspri = max;
        if (abspri < min) abspri = min;

      //  _pthr = [NSOperationQueue new];
      //  [_pthr addOperations:@[[NSBlockOperation blockOperationWithBlock:^{
      //      static_main(this);
      //  }]] waitUntilFinished:NO];
      
        parm.sched_priority = abspri;
        pthread_attr_init (&attr);
        pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
        pthread_attr_setschedpolicy (&attr, policy);
        pthread_attr_setschedparam (&attr, &parm);
        pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);
        pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setstacksize (&attr, 0x10000);
        pthread_create (&_pthr, &attr, static_main, this);
        pthread_attr_destroy (&attr);
        
      //  [NSThread detachNewThreadWithName:@"zitaConvolver" withPriority: abspri withBlock:^{
        //    static_main(this);
       // }];
    }

    void stop (void)
    {
        if (_stat != ST_IDLE) {
            _stat = ST_TERM;
            _trig.post ();
        }
    }

    void cleanup (void)
    {
        Inpnode *X, *X1;
        Outnode *Y, *Y1;
        Macnode *M, *M1;
        
        X = _inp_list;
        while (X)
        {
            X1 = X->_next;
            delete X;
            X = X1;
        }
        _inp_list = NULL;
        
        Y = _out_list;
        while (Y) {
            M = Y->_list;
            while (M) {
                M1 = M->_next;
                delete M;
                M = M1;
            }
            Y1 = Y->_next;
            delete Y;
            Y = Y1;
        }
        _out_list = NULL;
        _ir_list = NULL;
        
        _fftCycle = 0;
        _cycleTot = 0;
    }


    static void *static_main (void *arg)
    {
        ((Convlevel *) arg)->main();
        return 0;
    }

    void main (void)
    {
        _stat = ST_PROC;
        while (1)
        {
            _trig.wait ();
            if (_stat == ST_TERM) {
                if (_pthr) {
                    //[_pthr cancelAllOperations];
                    // [_pthr cancel];
                        pthread_join(_pthr, NULL);
                        pthread_cancel(_pthr);
                }
                _stat = ST_IDLE;
                _pthr = 0;
                return;
            }
            if constexpr(mode == USE_AVFFT)
                process_AVFFT (false);
            else
                process_PFFFT (false);
            _done.post ();
        }
    }

    void process_PFFFT(bool skip)
    {
        int         i, i1, j, k, n1, n2, opi1, opi2;
        cmplxT<T>   *fftb, *ffta;
        T           *inpd, *outd;
        
        i1 = _inpoffs;
        n1 = _parsize;
        n2 = 0;
        
        _inpoffs = i1 + n1;
        
        if (_inpoffs >= _inpsize) {
            _inpoffs -= _inpsize;
            n2 = _inpoffs;
            n1 -= n2;
        }
        opi1 = (_opind + 1) % 3;
        opi2 = (_opind + 2) % 3;
        
        for (auto X = _inp_list; X; X = X->_next)
        {
            inpd = _inpbuff [X->_inp];
            if (n1) {
                memcpy (_time_data, inpd + i1, n1 * sizeof (T));
            }
            if (n2){
                memcpy (_time_data + n1, inpd, n2 * sizeof (T));
            }
            memset (_time_data + _parsize, 0, _parsize * sizeof (T));
            if constexpr( OVERRIDE_PFFFT )
                _av.real_fft( (zfloat*)_time_data, (zfloat*)X->_ffta [_ptind], _parsize * 2, true );
            else {
                if ZFLOAT_IS_FLOAT
                    pffft_transform_ordered(_plan_r2c, (float*)_time_data, (float*)X->_ffta [_ptind], NULL, PFFFT_FORWARD);
                else
                    pffftd_transform_ordered(_d_plan_r2c, (double*)_time_data, (double*)X->_ffta [_ptind], NULL, PFFFT_FORWARD);
            }
            if constexpr( doUseVector )
                fftswap ((cmplxT<T> *)X->_ffta [_ptind]);
        }
        
        if (skip) {
            for (auto Y = _out_list; Y; Y = Y->_next) {
                memset ( Y->_buff [opi2], 0, _parsize * sizeof (T));
                memset ( Y->_buffo [opi2], 0, _parsize * sizeof (T));
            }
        } else
        {
            for (auto Y = _out_list; Y; Y = Y->_next)
            {
                memset (_freq_data, 0, (_parsize + 1) * sizeof (cmplxT<zfloat>));
                cmplxT<zfloat> *fd = (cmplxT<zfloat>*) _freq_data;
                
                for (auto M1 = Y->_list; M1; M1 = M1->_next)
                {
                    auto X1 = M1->_inpn;
                    i = _ptind;
                    for (j = 0; j < _npar; j++)
                    {
                        ffta = (cmplxT<T>*)X1->_ffta[i];
                        fftb = (cmplxT<T>*) ((M1->_link) ? M1->_link->_fftb [j] : M1->_fftb [j]);
                         //PRINT_D1(ffta[1].re);
                        // PRINT_D1(fftb[1].re);
                        if (fftb)
                        {
                            if constexpr( doUseVector )
                            {
                                FV4 *A = (FV4 *) ffta;
                                FV4 *B = (FV4 *) fftb;
                                FV4 *D = (FV4 *) _freq_data;
                                
                                for (k = 0; k < _parsize; k += 4) {
                                    D [0] += A [0] * B [0] - A [1] * B [1];
                                    D [1] += A [0] * B [1] + A [1] * B [0];
                                    A += 2; B += 2; D += 2;
                                }
                                cmplxT<zfloat> *fd = (cmplxT<zfloat>*) _freq_data;
                                fd [_parsize].re += ffta [_parsize].re * fftb [_parsize].re;
                                fd [_parsize].im = 0;
                            } else
                            {
                                for (k = 0; k <= _parsize; k++) {
                                    fd [k].re += ffta [k].re * fftb [k].re - ffta [k].im * fftb [k].im;
                                    fd [k].im += ffta [k].re * fftb [k].im + ffta [k].im * fftb [k].re;
                                }
                            }
                        }
                        if (i == 0)
                            i = _npar;
                        i--;
                    }
                }
                if constexpr( doUseVector )
                    fftswap ((cmplxT<T> *)_freq_data);
                if (_fftCycle >= 0)
                {
                    double scv = gVolume;
                    
                    cmplxT<T> *fd = _freq_data;
                    cmplxT<mssFloat2> *dLR = _spec_data;
                    
                    if (_fftCycle == _cycleTot)
                        memset(_spec_data, 0, _parsize * sizeof(cmplxT<mssFloat2>));
                    
                    if (_fftCycle & 1) 
                    {
                        for (int i=0; i<_parsize; i++)
                        {
                            cmplxT<T> si = fd[i] * scv;
                            dLR[i].re[1] += si.re;
                            dLR[i].im[1] += si.im;
                        }
                    } else 
                    {
                        for (int i=0; i<_parsize; i++)
                        {
                            cmplxT<T> si = fd[i] * scv;
                            dLR[i].re[0] += si.re;
                            dLR[i].im[0] += si.im;
                        }
                    }
                    if (_fftCycle == 0) {
                        specLock.lock();
                        gFFTQueue.WriteData(_spec_data, _parsize * sizeof(cmplxT<mssFloat2>));
                        specLock.unlock();
                    }
                    _fftCycle--;
                }
                if constexpr( OVERRIDE_PFFFT )
                    _av.real_fft( (zfloat*)_freq_data, (zfloat*)_time_data, _parsize * 2, false );
                else {
                    if ZFLOAT_IS_FLOAT
                        pffft_transform_ordered(_plan_c2r, (float*)_freq_data, (float*)_time_data, NULL, PFFFT_BACKWARD);
                    else
                        pffftd_transform_ordered(_d_plan_c2r, (double*)_freq_data, (double*)_time_data, NULL, PFFFT_BACKWARD);
                }
                outd = Y->_buff [opi1];
                
                for (k = 0; k < _parsize; k++)
                    outd [k] += _time_data [k];
                outd = Y->_buff [opi2];
                
                memcpy (outd, _time_data + _parsize, _parsize * sizeof (_time_data[0]));
            }
        }
        if (++_ptind == _npar)
            _ptind = 0;
    }
    
    void process_AVFFT (bool skip)
    {
        int             chn, row, i, i1, j, k, n1, n2, opi1, opi2;
        cmplxT<zfloat8> *fftb, *ffta, *fd;
        zfloat8         *outd, *td8 = (zfloat8*)_time_data;
        zfloat4         *inpd;
        
        i1 = _inpoffs;
        n1 = _parsize;
        n2 = 0;
        
        _inpoffs = i1 + n1;
        
        if (_inpoffs >= _inpsize) {
            _inpoffs -= _inpsize;
            n2 = _inpoffs;
            n1 -= n2;
        }
        opi1 = (_opind + 1) % 3;
        opi2 = (_opind + 2) % 3;
        
        inpd = _inpbuff_4;
        
        for (auto M1 = _ir_list; M1; M1 = M1->_next)
        {
            zfloat4 s;
            int8v mask = M1->_inpmask;
            chn = M1->_chn;
            row = ceil((zfloat)( chn + 1) / 8.) - 1;
            
            ffta = &((cmplxT<zfloat8>*) M1->_inpn->_ffta[_ptind])[row * (_parsize + 1)];
            if (!((chn+1) % 8))
            {
                if (n1) {
                    for (int i = 0; i < n1; i++) {
                        s = inpd[i1+i];
                        td8[i] = shufflevector( make_zfloat8(s, s), mask );
                    }
                }
                if (n2) {
                    for (int i = 0; i < n2; i++) {
                        s = inpd[i];
                        td8[i] = shufflevector( make_zfloat8(s, s), mask );
                    }
                }
                memset(td8 + _parsize, 0, _parsize * sizeof(zfloat8));
                _av8.real_fft( td8, (zfloat8*) ffta, 2 * _parsize, true );
            }
        }
        if (skip) {
            Outnode *Y = _out_list;
            memset( Y->_buff [opi2], 0, _parsize * sizeof (zfloat8) );
            memset( Y->_buffo [opi2], 0, _parsize * sizeof (zfloat8) );
        } else
        {
            
            auto Y = _out_list;
            memset(_freq_data, 0, (_parsize + 1) * sizeof (cmplxT<zfloat8>));
            
            for (auto M1 = _ir_list; M1; M1 = M1->_next)
            {
                auto X1 = M1->_inpn;
                i = _ptind;
                chn = M1->_chn;
                row = ceil((zfloat)( chn + 1) / 8.) - 1;
                
                cmplxT<zfloat8>** _ffta = (cmplxT<zfloat8>**) X1->_ffta;
                cmplxT<zfloat8>** _fftb = (cmplxT<zfloat8>**) M1->_fftb;
                
                int8v mask = M1->_outmask;
                cmplxT<zfloat8> neg = cmplxT<zfloat8>(M1->_outneg, M1->_outneg);
                
                int rowpar = row * (_parsize + 1);
                
                if (!(chn % 8)) {
                    for (j = 0; j < _npar; j++)
                    {
                        fd = (cmplxT<zfloat8>*) _freq_data;
                        ffta = &_ffta[i][rowpar];
                        
                        fftb = _fftb[j];
                        for (k = 0; k <= _parsize; k++) {
                            *fd++ += (neg * (*ffta++ ^ *fftb++)) & mask;
                        }
                        if (i == 0) i = _npar;
                        i--;
                    }
                }
            }
            
            _av8.real_fft( (zfloat8*)_freq_data, (zfloat8*)_time_data, _parsize * 2, false );
            outd = Y->_buff[opi1];
            
            for (k = 0; k < _parsize; k++)
                outd[k] += _time_data[k];
            outd = Y->_buff[opi2];
            
            memcpy( outd, _time_data + _parsize, _parsize * sizeof (zfloat8) );
            
            memset( _spec_data, 0, _parsize * sizeof(cmplxT<mssFloat2>) );
            for (int ch=0; ch<8; ch++)
            {
                double scv = gVolume;
                cmplxT<zfloat8> *fd = _freq_data;
                cmplxT<mssFloat2> *dLR = _spec_data;
                
                for (int i=0; i<_parsize; i++)
                {
                    cmplxT<zfloat8> si = fd[i] * scv;
                    dLR[i].re[ch&1] += si.re[ch];
                    dLR[i].im[ch&1] += si.im[ch];
                }
            }
            specLock.lock();
            gFFTQueue.WriteData(_spec_data, _parsize * sizeof(cmplxT<mssFloat2>));
            specLock.unlock();
        }
        if (++_ptind == _npar)
            _ptind = 0;
    }

    int readout( bool sync, long skipcnt, bool eq_active )
    {
        int i;
        
        _eq_active = eq_active;
        _outoffs += _outsize;
        if (_outoffs == _parsize)
        {
            _outoffs = 0;
            if (_stat == ST_PROC)
            {
                while (_wait)
                {
                    if (sync) _done.wait ();
                    else if (_done.trywait ())
                        break;
                    _wait--;
                }
                if (++_opind == 3) _opind = 0;
                _trig.post ();
                _wait++;
            } else
            {
                if constexpr(mode == USE_AVFFT)
                    process_AVFFT( skipcnt >= 2 * _parsize );
                else
                    process_PFFFT( skipcnt >= 2 * _parsize );
                
                if (++_opind == 3)
                    _opind = 0;
            }
        }
        if constexpr(mode == USE_AVFFT)
        {
            zfloat8  *p, *q;
            
            auto Y = _out_list;
            //for (; Y; Y = Y->_next) {
            p = ((zfloat8*)Y->_buff [_opind]) + _outoffs;
            q = _outbuff_8; // [Y->_out];
            for (i = 0; i < _outsize; i++)
                q[i] += p [i];
           // }
        } else {  // PFFFT
            T   *p, *q;
            for (auto Y = _out_list; Y; Y = Y->_next) {
                p = Y->_buff [_opind] + _outoffs;
                q = _outbuff [Y->_out];
                for (i = 0; i < _outsize; i++)
                    q[i] += p [i];
            }
        }
        return (_wait > 1) ? _bits : 0;
    }

    void print (FILE *F)
    {
        //    LOGGER ("prio = %4d, offs = %6d,  parsize = %5d,  npar = %3d\n", _prio, _offs, _parsize, _npar);
    }
    
    Macnode *findmacnode_PFFFT (int inp, int out, bool create)
    {
        Inpnode *X, *XR;
        Outnode *Y, *YR;
        Macnode *M;
    
        for (X = _inp_list; X && (X->_inp != inp); X = X->_next) ;
        
        if (! X)
        {
            if (! create)
                return 0;
            X = new Inpnode (inp);
            X->_next = _inp_list;
            _inp_list = X;
            X->alloc_ffta_pffft (_npar, _parsize * 2);
        }
        
        for (Y = _out_list; Y && (Y->_out != out); Y = Y->_next) ;
        if (!Y)
        {
            if (! create) return 0;
            Y = new Outnode (out, _parsize);
            Y->_next = _out_list;
            _out_list = Y;
        }
        
        for (M = Y->_list; M && (M->_inpn != X); M = M->_next) ;
        if ( !M ) {
            if ( !create )
                return 0;
            M = new Macnode(X);
            M->_next = Y->_list;
            Y->_list = M;
        }

        return M;
    }
    
    Macnode *findmacnode_AVFFT (int chn, int inp, int out, bool create)
    {
        Inpnode *X, *XR;
        Outnode *Y, *YR;
        Macnode *M;
        
        XR = _inp_list;
        for (X = _inp_list; X && (X->_inp != inp); X = X->_next) ;
        if (! X)
        {
            if (! create)
                return 0;
            X = new Inpnode (inp);
            
            X->_next = _inp_list;
            _inp_list = X;
            if (XR) {
                X->link_ffta(XR); 
            } else
                X->alloc_ffta_avfft (_npar, _parsize * 2);
        }
        YR = _out_list;
        for (Y = _out_list; Y && (Y->_out != out); Y = Y->_next) ;
        if (! Y)
        {
            if (! create) return 0;
            Y = (YR) ? new Outnode (out, YR) : new Outnode (out, 2 * _parsize);
            Y->_next = _out_list;
            _out_list = Y;
        }
        
        if (_ir_list == NULL) {
            M = new Macnode(X);
            M->_chn = chn;
            _ir_list = M;
        } else {
            for (M = _ir_list; M->_next; M = M->_next) ;
            Macnode *next = new Macnode(X);
            next->_chn = chn;
            M = M->_next = next;
        }
        return M;
    }
    
    void fftswap (cmplxT<T> *p)
    {
        long  n = _parsize;
        T     a, b;
        
        while (n)
        {
            a = p [2].re;
            b = p [3].re;
            p [2].re = p [0].im;
            p [3].re = p [1].im;
            p [0].im = a;
            p [1].im = b;
            
            p += 4;
            n -= 4;
        }
    }
    
    
    int setimpmap(int ch, int inp) {
        _inpmap[ch] = inp;
    }
protected:
    
    Macnode          *_mlink;
    int              _inpmap[MAXINP]; // inputs map
    volatile long    _stat;           // current processing state
    int              _cycleTot;       // cycles total
    int              _fftCycle;       // cycle variable for sync
    int              _nch;            // number of total IR channels
    int              _prio;           // relative priority
    int              _offs;           // offset from start of impulse response
    int              _npar;           // number of partitions
    int              _parsize;        // partition and outbut buffer size
    int              _outsize;        // step size for output buffer
    int              _outoffs;        // offset into output buffer
    int              _inpsize;        // size of shared input buffer
    int              _inpoffs;        // offset into input buffer
    int              _options;        // various options
    int              _ptind;          // rotating partition index
    int              _opind;          // rotating output buffer index
    int              _bits;           // bit identifiying this level
    int              _wait;           // number of unfinished cycles
    //NSOperationQueue *_pthr;
    //NSThread      *_pthr;
    pthread_t        _pthr;          // posix thread executing this level
    ZCsema           _trig;          // sema used to trigger a cycle
    ZCsema           _done;          // sema used to wait for a cycle
    Inpnode *_inp_list;      // linked list of active inputs
    Outnode *_out_list;      // linked list of active outputs
    Macnode *_ir_list;       // link node that gets linked by chain
    cmplxT<T>       *_freq_data;     // workspace
    PFFFT_Setup     *_plan_r2c;      // FFTW plan, forward FFT
    PFFFT_Setup     *_plan_c2r;      // FFTW plan, inverse FFT
    PFFFTD_Setup    *_d_plan_r2c;    // FFTW plan, forward FFT
    PFFFTD_Setup    *_d_plan_c2r;    // FFTW plan, inverse FFT
    T               *_time_data;     // workspace
    zfloat          *_prep_data;     // workspace
    zfloat4         *_inpbuff_4;     // 4 channel in
    zfloat8         *_outbuff_8;     // 8 chanel out
    T              **_inpbuff;       // array of shared input buffers
    T              **_outbuff;       // array of shared output buffers
    cmplxT<mssFloat2> *_spec_data;   // spectrum computation
    T               *_win;           // hanning window
    AVFFT<zfloat>   _av;             // zfloat fft
    AVFFT<zfloat8>  _av8;            // 8 x zfloat fft
    
    FFTReal<zfloat> *_fft;           // zfloat fft
    FFTReal<zfloat8> *_fft8;         // 8 x zfloat fft
    
    bool            _eq_active;     // eq is active
};

template <typename T, FFT_MODE mode, bool OVERRIDE_PFFFT>
class Convproc
{
    friend class Convlevel<T, mode, OVERRIDE_PFFFT>;
public:
    enum {
        ST_IDLE = 0,
        ST_STOP,
        ST_WAIT,
        ST_PROC
    };
    enum {
        FL_LATE = 0x0000FFFF,
        FL_LOAD = 0x01000000
    };
    enum  {
        OPT_FFTW_MEASURE = Convlevel<T, mode, OVERRIDE_PFFFT>::OPT_FFTW_MEASURE,
        OPT_VECTOR_MODE  = Convlevel<T, mode, OVERRIDE_PFFFT>::OPT_VECTOR_MODE,
        OPT_LATE_CONTIN  = Convlevel<T, mode, OVERRIDE_PFFFT>::OPT_LATE_CONTIN
    };
   Convproc () :
    _state (ST_IDLE),
    _options (OPT_VECTOR_MODE),
    _skipcnt (0),
    _ninp (0),
    _nout (0),
    _quantum (0),
    _minpart (0),
    _maxpart (0),
    _nlevels (0),
    _latecnt (0),
    _inpbuff_4 (0),
    _outbuff_8 (0)
    {
        _mac_cost = DEFAULT_MAC_COST;
        _fft_cost = DEFAULT_FFT_COST;

        if constexpr( mode == USE_PFFFT ) {
            memset (_inpbuff, 0, MAXINP * sizeof (T *));
            memset (_outbuff, 0, MAXOUT * sizeof (T *));
        }
        memset (_convlev, 0, MAXLEV * sizeof (Convlevel<T, mode, OVERRIDE_PFFFT> *));
    }

    ~Convproc (void)
    {
        stop_process ();
        cleanup ();
        
        for (int k = 0; k < _nlevels; k++)
            if (_convlev[k]) {
                delete _convlev[k];
                _convlev [k] = NULL;
            }
    }

    void set_options (int options)
    {
        _options = options;
    }

    void set_skipcnt (long skipcnt)
    {
        if ((_quantum == _minpart) && (_quantum == _maxpart))
            _skipcnt = skipcnt;
    }

    int configure (int nch, int ninp, int nout, int maxsize, int quantum, int minpart, int maxpart, float density)
    {
        int     offs, npar, size, pind, nmin, i;
        int     prio, step, d, r, s;
        float   cfft, cmac;
        
        if (_state != ST_IDLE) {
            fprintf(stderr, "Converror::BAD_STATE\n");
            return Converror::BAD_STATE;
        }
        if (   (ninp < 1) || (ninp > MAXINP)
            || (nout < 1) || (nout > MAXOUT)
            || (quantum & (quantum - 1))
            || (quantum < MINQUANT)
            || (quantum > MAXQUANT)
            || (minpart & (minpart - 1))
            || (minpart < MINPART)
            || (minpart < quantum)
            || (minpart > MAXDIVIS * quantum)
            || (maxpart & (maxpart - 1))
            || (maxpart > MAXPART)
            || (maxpart < minpart)) {
            fprintf(stderr, "Converror::BAD_PARAM\n");
            return Converror::BAD_PARAM;
        }
        
        nmin = (ninp < nout) ? ninp : nout;
        if (density <= 0.0f) density = 1.0f / nmin;
        if (density >  1.0f) density = 1.0f;
        cfft = _fft_cost * (ninp + nout);
        cmac = _mac_cost * ninp * nout * density;
        step = (cfft < 4 * cmac) ? 1 : 2;
        if (step == 2)
        {
            r = (int)maxpart / minpart;
            s = (r & 0xAAAA) ? 1 : 2;
        }
        else s = 1;
        nmin = (s == 1) ? 2 : 8; // d: : 6
        if (minpart == quantum)
            nmin++;
        prio = 0;
        size = quantum;
        while (size < minpart)
        {
            prio -= 1;
            size <<= 1;
        }
        
        try {
            for (offs = pind = 0; offs < maxsize; pind++)
            {
                npar = (maxsize - offs + size - 1) / size;
                if ((size < maxpart) && (npar > nmin))
                {
                    r = 1 << s;
                    d = npar - nmin;
                    d = d - (d + r - 1) / r;
                    if (cfft < d * cmac)
                        npar = nmin;
                }
                _convlev [pind] = new Convlevel<T, mode, OVERRIDE_PFFFT> ();
                _convlev [pind]->configure (nch, prio, offs, npar, size, _options);
                offs += size * npar;
                if (offs < maxsize)
                {
                    prio -= s;
                    size <<= s;
                    s = step;
                    nmin = (s == 1) ? 2 : 8; // d: : 6
                }
            }
            
            _ninp = ninp;
            _nout = nout;
            _quantum = quantum;
            _minpart = minpart;
            _maxpart = size;
            _nlevels = pind;
            _latecnt = 0;
            _inpsize = 2 * size;
            
            if constexpr( mode == USE_AVFFT ) {
                _inpbuff_4 = callocT<zfloat4> (_inpsize); THROW_MEM(_inpbuff_4 == NULL);
                _outbuff_8 = callocT<zfloat8> (_minpart); THROW_MEM(_outbuff_8 == NULL);
            } else {
                for (i = 0; i < ninp; i++) {
                    _inpbuff [i] = callocT<T> (_inpsize); THROW_MEM(_inpbuff [i] == NULL);
                }
                for (i = 0; i < nout; i++) {
                    _outbuff [i] = callocT<T> (_minpart); THROW_MEM(_outbuff [i] == NULL);
                }
            }
        } catch (...)
        {
            cleanup ();
            LOG("Converror::MEM_ALLOC");
            return Converror::MEM_ALLOC;
        }
        _state = ST_STOP;
        return 0;
    }
    
    int impdata_create (int c, int *inp, int *out, int step, float *data, int ind0, int ind1, int nch)
    {
        long j;
        
        _nch = nch;
        if (_state != ST_STOP) return Converror::BAD_STATE;
        if ((inp[c]-1 >= _ninp) || (out[c]-1 >= _nout))
            return Converror::BAD_PARAM;
        try {
            for (j = 0; j < _nlevels; j++) {
                if constexpr( mode == USE_PFFFT)
                    _convlev [j]->impdata_write_PFFFT (c, inp, out, step, data, ind0, ind1, nch, true);
                else
                    _convlev [j]->impdata_write_AVFFT (c, inp, out, step, data, ind0, ind1, nch, true);
            }
        } catch (...)
        {
            fprintf(stderr, "impdata_create() gave Converror::MEM_ALLOC!\n");
            cleanup ();
            return Converror::MEM_ALLOC;
        }
        return 0;
    }

    int impdata_clear (int inp, int out)
    {
        long k;
        
        if (_state < ST_STOP)
            return Converror::BAD_STATE;
        for (k = 0; k < _nlevels; k++)
            _convlev [k]->impdata_clear (inp, out);
        return 0;
    }

    int impdata_update (int inp, int out, int step, float *data, int ind0, int ind1)
    {
        long j;
        
        if (_state < ST_STOP)
            return Converror::BAD_STATE;
        
        if ((inp >= _ninp) || (out >= _nout))
            return Converror::BAD_PARAM;
        
        for (j = 0; j < _nlevels; j++) {
            _convlev [j]->impdata_write (inp, out, step, data, ind0, ind1, _nch, false);
        }
        return 0;
    }

    int impdata_link (int inp1, int out1, int inp2, int out2)
    {
        long j;
        
        if ((inp1 >= _ninp) || (out1 >= _nout))
            return Converror::BAD_PARAM;
        if ((inp2 >= _ninp) || (out2 >= _nout))
            return Converror::BAD_PARAM;
        if ((inp1 == inp2) && (out1 == out2))
            return Converror::BAD_PARAM;
        if (_state != ST_STOP)
            return Converror::BAD_STATE;
        try {
            for (j = 0; j < _nlevels; j++)
                _convlev [j]->impdata_link (inp1, out1, inp2, out2);
        } catch (...)
        {
            cleanup ();
            return Converror::MEM_ALLOC;
        }
        return 0;
    }


    int reset ()
    {
        int k;
      //  LOG("reset");
        if (_state == ST_IDLE) {
           LOG("reset() from a bad state: %d (!ST_IDLE)", _state);
            return Converror::BAD_STATE;
        }
        if constexpr(mode == USE_AVFFT)
        {
            if (_inpbuff_4)
                memset (_inpbuff_4, 0, _inpsize * sizeof (zfloat4));
            if (_outbuff_8)
                memset (_outbuff_8, 0, _minpart * sizeof (zfloat8));
            if (_convlev [0]) _convlev [0]->reset (_inpsize, _minpart, _inpbuff_4, _outbuff_8);
        } else {
            for (k = 0; k < _ninp; k++)
                if (_inpbuff [k])
                    memset (_inpbuff [k], 0, _inpsize * sizeof (T));
            for (k = 0; k < _nout; k++)
                if (_outbuff [k])
                    memset (_outbuff [k], 0, _minpart * sizeof (T));
            for (k = 0; k < _nlevels; k++)
                _convlev [k]->reset (_inpsize, _minpart, _inpbuff, _outbuff);
        }
        return 0;
    }

    int start_process (int abspri, int policy)
    {
        long k;
        
      //  if (_state != ST_STOP)
        //    return Converror::BAD_STATE;
        _latecnt = 0;
        _inpoffs = 0;
        _outoffs = 0;
        reset ();
        
        for (k = (_minpart == _quantum) ? 1 : 0; k < _nlevels; k++)
            _convlev [k]->start (abspri, policy);
        _state = ST_PROC;
        
        return 0;
    }


    int process( bool sync, bool eq_active )
    {
        int k, f = 0;
        
        if (_state != ST_PROC)
            return 0;
        
        _inpoffs += _quantum;
        if (_inpoffs == _inpsize)
            _inpoffs = 0;
        _outoffs += _quantum;
        
        for (k = 0; k < _nlevels; k++) {
            int total = std::min(_nch,_nout)-1;
            _convlev [k]->_fftCycle = total;
            _convlev [k]->_cycleTot = total;
        }
        
        if (_outoffs == _minpart)
        {
            _outoffs = 0;
            
            if constexpr(mode == USE_PFFFT) {
                for (k = 0; k < _nout; k++)
                    memset (_outbuff [k], 0, _minpart * sizeof (T));
            } else {
                memset (_outbuff_8, 0, _minpart * sizeof (zfloat8));
            }
           
            for (k = 0; k < _nlevels; k++)
                f |= _convlev [k]->readout( sync, _skipcnt,  eq_active );
            if (_skipcnt < _minpart)
                _skipcnt = 0;
            else
                _skipcnt -= _minpart;
            if (f) {
                if (++_latecnt >= 5) {
                    if (~_options & OPT_LATE_CONTIN)
                        stop_process ();
                    f |= FL_LOAD;
                }
            } else
                _latecnt = 0;
        }
        return f;
    }


    int stop_process (void)
    {
        long k;
        
        if (_state != ST_PROC)
            return Converror::BAD_STATE;
        for (k = 0; k < _nlevels; k++)
            _convlev [k]->stop ();
        _state = ST_WAIT;
        
        return 0;
    }


    int cleanup (void)
    {
        int k;
        
        while (! check_stop ()) pthread_yield();
        if constexpr( mode == USE_AVFFT )
        {
             callocT_free(_inpbuff_4); _inpbuff_4 = 0;
             callocT_free(_outbuff_8); _outbuff_8 = 0;
        } else
        {
            for (k = 0; k < _ninp; k++) {
                callocT_free(_inpbuff [k]); _inpbuff [k] = 0;
            }
            for (k = 0; k < _nout; k++) {
                callocT_free(_outbuff [k]); _outbuff [k] = 0;
            }
        }
        _state = ST_IDLE;
        _options = OPT_VECTOR_MODE;
        _skipcnt = 0;
        _ninp = 0;
        _nout = 0;
        _quantum = 0;
        _minpart = 0;
        _maxpart = 0;
        _nlevels = 0;
        _latecnt = 0;
        
        return 0;
    }


    bool check_stop (void)
    {
        long k;
        
        for (k = 0; (k < _nlevels) && (_convlev [k]->_stat == ST_IDLE); k++);
        if (k == _nlevels) {
            _state = ST_STOP;
            return true;
        }
        return false;
    }

    void print (FILE *F)
    {
        for (int k = 0; k < _nlevels; k++)
            _convlev [k]->print (F);
    }
    
    long state (void) const {
        return _state;
    }

    zfloat4 *inpdata () const {
        return (_inpbuff_4) ? _inpbuff_4 + _inpoffs : NULL; // [inp] + _inpoffs;
    }
    
    zfloat8 *outdata () const {
        return (_outbuff_8) ? _outbuff_8 + _outoffs : NULL; //[out] + _outoffs;
    }

    T *inpdata (int inp) const {
        return _inpbuff [inp] + _inpoffs;
    }
    
    T *outdata (int out) const {
        return _outbuff [out] + _outoffs;
    }
    
protected:
    
    int      _state;                  // current state
    zfloat4  *_inpbuff_4;
    zfloat8  *_outbuff_8;
    
    T       *_inpbuff [MAXINP];       // input buffers
    T       *_outbuff [MAXOUT];       // output buffers

    int     _inpoffs;                 // current offset in input buffers
    int     _outoffs;                 // current offset in output buffers
    int     _options;                 // option bits
    int     _skipcnt;                 // number of frames to skip
    int     _ninp;                    // number of inputs
    int     _nout;                    // number of outputs
    int     _nch;                     // number of impulse channels
    int     _quantum;                 // processing block size
    int     _minpart;                 // smallest partition size
    int     _maxpart;                 // largest allowed partition size
    int     _nlevels;                 // number of partition sizes
    int     _inpsize;                 // size of input buffers
    int     _latecnt;                 // count of cycles ending too late
    Convlevel<T, mode, OVERRIDE_PFFFT> *_convlev [MAXLEV];    // array of processors
     float  _mac_cost;
     float  _fft_cost;
};


