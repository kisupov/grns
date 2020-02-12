/*
 * specAddition.h
 *
 * This file is part of CAMPARY Library
 *
 * Copyright (C) 2014 - 
 *
 * CAMPARY Library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * CAMPARY Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MultiPrecGPU Library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, 
 * Boston, MA  02110-1301  USA
 */
 
/* Contributors: Valentina Popescu valentina.popescu@ens-lyon.fr
 *               Mioara Joldes joldes@laas.fr
 */

#ifndef _specAddition_h
#define _specAddition_h

/****************************************************/
template <>
inline void merge<1,1>(double const *x, double const *y, double *z){
  if(fabs(y[0])>fabs(x[0])){ z[0] = y[0]; z[1] = x[0]; } else { z[0] = x[0]; z[1] = y[0]; }
}

/****************************************************/
template <>
inline void certifiedAdd<1,1,1>(const double *x, const double *y, double *r){ r[0] = FPadd_rn(x[0], y[0]); }
template <>
inline void certifiedAdd<1,1,2>(const double *x, const double *y, double *r){ r[0] = two_sum(x[0], y[0], r[1]); }
template <>
inline void certifiedAdd<1,2,2>(const double *x, const double *y, double *r){ 
// DWPlusFP with relative error <= 2u^2 + 5u^3
  double s, e;
  s = two_sum( x[0], y[0], e );
  e = FPadd_rn( y[1], e );
  r[0] = fast_two_sum( s, e, r[1] );
}
template <>
inline void certifiedAdd<2,1,2>(const double *x, const double *y, double *r){
// DWPlusFP with relative error <= 3u^2 + 13u^3
  double s, e;
  s = two_sum( x[0], y[0], e );
  e = FPadd_rn( x[1], e );
  r[0] = fast_two_sum( s, e, r[1] );
}
template<>
inline void certifiedAdd<2,2,2>(double const *x, double const *y, double *r){
// AccurateDWPlusDW with relative error <= 2u^2 + 5 u^3
  double sh, sl, th, tl;
  sh = two_sum( x[0], y[0], sl );
  th = two_sum( x[1], y[1], tl );
  sl = FPadd_rn( sl, th );
  sh = fast_two_sum( sh, sl, sl);
  sl = FPadd_rn( tl, sl );
  r[0] = fast_two_sum( sh, sl, r[1] );
}

/****************************************************/
template <>
inline void baileyAdd_fast<1,1,1>(const double *x, const double *y, double *z){ z[0] = FPadd_rn(x[0], y[0]); }
template <>
inline void baileyAdd_fast<1,1,2>(const double *x, const double *y, double *z){ z[0] = two_sum(x[0], y[0], z[1]); }
template <>
inline void baileyAdd_fast<1,2,2>(const double *x, const double *y, double *z){
// DWPlusFP with relative error <= 2u^2 + 5u^3
  double s, e;
  s = two_sum( x[0], y[0], e );
  e = FPadd_rn( y[1], e );
  z[0] = fast_two_sum( s, e, z[1] );
}
template <>
inline void baileyAdd_fast<2,1,2>(const double *x, const double *y, double *z){
// DWPlusFP with relative error <= 2u^2 + 5u^3
  double s, e;
  s = two_sum( x[0], y[0], e );
  e = FPadd_rn( x[1], e );
  z[0] = fast_two_sum( s, e, z[1] );
}
template<>
inline void baileyAdd_fast<2,2,2>(double const *x, double const *y, double *z){
// SloppyDWPlusDW with relative error up to 1 --> untrusted results
  double sh, sl, v;
  sh = two_sum( x[0], y[0], sl );
  v = FPadd_rn( x[1], y[1]);
  v = FPadd_rn( sl, v );
  z[0] = fast_two_sum( sh, v, z[1] );
}
template<>
inline void baileyAdd_fast<3,1,3>(double const *x, double const *y, double *z){
  renorm2L_4Add1<3,3>( x, y[0], z );
}
template<>
inline void baileyAdd_fast<3,3,3>(double const *x, double const *y, double *z){
	double e, f[4];
  f[2] = two_sum ( x[2], y[2], f[3] );
  f[1] = two_sum ( x[1], y[1], e );
  f[2] = two_sum ( f[2], e,    e );
  f[3] = FPadd_rn( f[3], e );
  f[0] = two_sum ( x[0], y[0], e );
  f[1] = two_sum ( f[1], e,    e );
  f[2] = two_sum ( f[2], e,    e );
  f[3] = FPadd_rn( f[3], e );
  fast_renorm2L<4,3>( f, z );
}
template<>
inline void baileyAdd_fast<4,1,4>(double const *x, double const *y, double *z){
  renorm2L_4Add1<4,4>( x, y[0], z );
}
template<>
inline void baileyAdd_fast<4,4,4>(double const *x, double const *y, double *z){
  double e, f[5];
  f[3] = two_sum ( x[3], y[3], f[4] );
  f[2] = two_sum ( x[2], y[2], e );
  f[3] = two_sum ( f[3], e,    e );
  f[4] = FPadd_rn( f[4], e );
  f[1] = two_sum ( x[1], y[1], e );
  f[2] = two_sum ( f[2], e,    e );
  f[3] = two_sum ( f[3], e,    e );
  f[4] = FPadd_rn( f[4], e );
  f[0] = two_sum ( x[0], y[0], e );
  f[1] = two_sum ( f[1], e,    e );
  f[2] = two_sum ( f[2], e,    e );
  f[3] = two_sum ( f[3], e,    e );
  f[4] = FPadd_rn( f[4], e );
  fast_renorm2L<5,4>( f, z );
}

template<>
inline void baileyAdd_fast<5,1,5>(double const *x, double const *y, double *z){
  renorm2L_4Add1<5,5>( x, y[0], z );
}

template<>
inline void baileyAdd_fast<5,5,5>(double const *x, double const *y, double *z){
	double e, f[6];
	f[5] = 0.0;
	f[4] = two_sum(x[4], y[4], e);
	f[5] = FPadd_rn(f[5], e);
	f[3] = two_sum(x[3], y[3], e);
	f[4] = two_sum(f[4], e, e);
	f[5] = FPadd_rn(f[5], e);
	f[2] = two_sum(x[2], y[2], e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = FPadd_rn(f[5], e);
	f[1] = two_sum(x[1], y[1], e);
	f[2] = two_sum(f[2], e, e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = FPadd_rn(f[5], e);
	f[0] = two_sum(x[0], y[0], e);
	f[1] = two_sum(f[1], e, e);
	f[2] = two_sum(f[2], e, e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = FPadd_rn(f[5], e);
  fast_renorm2L<6,5>( f, z );
}

template<>
inline void baileyAdd_fast<6,1,6>(double const *x, double const *y, double *z){
  renorm2L_4Add1<6,6>( x, y[0], z );
}

template<>
inline void baileyAdd_fast<6,6,6>(double const *x, double const *y, double *z){
	double e, f[7];
	f[6] = 0.0;
	f[5] = two_sum(x[5], y[5], e);
	f[6] = FPadd_rn(f[6], e);
	f[4] = two_sum(x[4], y[4], e);
	f[5] = two_sum(f[5], e, e);
	f[6] = FPadd_rn(f[6], e);
	f[3] = two_sum(x[3], y[3], e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = FPadd_rn(f[6], e);
	f[2] = two_sum(x[2], y[2], e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = FPadd_rn(f[6], e);
	f[1] = two_sum(x[1], y[1], e);
	f[2] = two_sum(f[2], e, e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = FPadd_rn(f[6], e);
	f[0] = two_sum(x[0], y[0], e);
	f[1] = two_sum(f[1], e, e);
	f[2] = two_sum(f[2], e, e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = FPadd_rn(f[6], e);
	fast_renorm2L<7,6>( f, z );
}

template<>
inline void baileyAdd_fast<7,1,7>(double const *x, double const *y, double *z){
  renorm2L_4Add1<7,7>( x, y[0], z );
}

template<>
inline void baileyAdd_fast<7,7,7>(double const *x, double const *y, double *z){
	double e, f[8];
	f[7] = 0.0;
	f[6] = two_sum(x[6], y[6], e);
	f[7] = FPadd_rn(f[7], e);
	f[5] = two_sum(x[5], y[5], e);
	f[6] = two_sum(f[6], e, e);
	f[7] = FPadd_rn(f[7], e);
	f[4] = two_sum(x[4], y[4], e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = FPadd_rn(f[7], e);
	f[3] = two_sum(x[3], y[3], e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = FPadd_rn(f[7], e);
	f[2] = two_sum(x[2], y[2], e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = FPadd_rn(f[7], e);
	f[1] = two_sum(x[1], y[1], e);
	f[2] = two_sum(f[2], e, e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = FPadd_rn(f[7], e);
	f[0] = two_sum(x[0], y[0], e);
	f[1] = two_sum(f[1], e, e);
	f[2] = two_sum(f[2], e, e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = FPadd_rn(f[7], e);
	fast_renorm2L<8,7>( f, z );
}

template<>
inline void baileyAdd_fast<8,1,8>(double const *x, double const *y, double *z){
  renorm2L_4Add1<8,8>( x, y[0], z );
}

template<>
inline void baileyAdd_fast<8,8,8>(double const *x, double const *y, double *z){
	double e, f[9];
	f[8] = 0.0;
	f[7] = two_sum(x[7], y[7], e);
	f[8] = FPadd_rn(f[8], e);
	f[6] = two_sum(x[6], y[6], e);
	f[7] = two_sum(f[7], e, e);
	f[8] = FPadd_rn(f[8], e);
	f[5] = two_sum(x[5], y[5], e);
	f[6] = two_sum(f[6], e, e);
	f[7] = two_sum(f[7], e, e);
	f[8] = FPadd_rn(f[8], e);
	f[4] = two_sum(x[4], y[4], e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = two_sum(f[7], e, e);
	f[8] = FPadd_rn(f[8], e);
	f[3] = two_sum(x[3], y[3], e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = two_sum(f[7], e, e);
	f[8] = FPadd_rn(f[8], e);
	f[2] = two_sum(x[2], y[2], e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = two_sum(f[7], e, e);
	f[8] = FPadd_rn(f[8], e);
	f[1] = two_sum(x[1], y[1], e);
	f[2] = two_sum(f[2], e, e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = two_sum(f[7], e, e);
	f[8] = FPadd_rn(f[8], e);
	f[0] = two_sum(x[0], y[0], e);
	f[1] = two_sum(f[1], e, e);
	f[2] = two_sum(f[2], e, e);
	f[3] = two_sum(f[3], e, e);
	f[4] = two_sum(f[4], e, e);
	f[5] = two_sum(f[5], e, e);
	f[6] = two_sum(f[6], e, e);
	f[7] = two_sum(f[7], e, e);
	f[8] = FPadd_rn(f[8], e);
	fast_renorm2L<9,8>( f, z );
}


template<>
inline void baileyAdd_fast<3,2,2>(double const *x, double const *y, double *z){
  double e, f[3];
  f[1] = two_sum ( x[1], y[1], e );
  f[2] = FPadd_rn( x[2], e );
  f[0] = two_sum ( x[0], y[0], e );
  f[1] = two_sum ( f[1], e,    e );
  f[2] = FPadd_rn( f[2], e );
  fast_renorm2L<3,2>( f, z );
}
template<>
inline void baileyAdd_fast<3,2,3>(double const *x, double const *y, double *z){
  double e, f[4];
  f[1] = two_sum ( x[1], y[1], e );
  f[2] = two_sum ( x[2], e,    f[3] );
  f[0] = two_sum ( x[0], y[0], e );
  f[1] = two_sum ( f[1], e,    e );
  f[2] = two_sum ( f[2], e,    e );
  f[3] = FPadd_rn( f[3], e );
  fast_renorm2L<4,3>( f, z );
}

template<>
inline void baileyAdd_fast<4,2,2>(double const *x, double const *y, double *z){
  double e, f[3];
  f[1] = two_sum ( x[1], y[1], e );
  f[2] = FPadd_rn( x[2], e );
  f[0] = two_sum ( x[0], y[0], e );
  f[1] = two_sum ( f[1], e,    e );
  f[2] = FPadd_rn( f[2], e );
  fast_renorm2L<3,2>( f, z );
}
template<>
inline void baileyAdd_fast<4,2,4>(double const *x, double const *y, double *z){
  double e, f[5];
  f[1] = two_sum ( x[1], y[1], e );
  f[2] = two_sum ( x[2], e,    e );
  f[3] = two_sum ( x[3], e,    f[4] );
  f[0] = two_sum ( x[0], y[0], e );
  f[1] = two_sum ( f[1], e,    e );
  f[2] = two_sum ( f[2], e,    e );
  f[3] = two_sum ( f[3], e,    e );
  f[4] = FPadd_rn( f[4], e );
  fast_renorm2L<5,4>( f, z );
}

/****************************************************/
template <>
inline void baileyAdd_renorm<1,1,1>(const double *x, const double *y, double *z){ z[0] = FPadd_rn(x[0], y[0]); }
template <>
inline void baileyAdd_renorm<1,1,2>(const double *x, const double *y, double *z){ z[0] = two_sum(x[0], y[0], z[1]); }
template <>
inline void baileyAdd_renorm<1,2,2>(const double *x, const double *y, double *z){
// DWPlusFP with relative error <= 2u^2 + 5u^3
  double s, e;
  s = two_sum( x[0], y[0], e );
  e = FPadd_rn( y[1], e );
  z[0] = fast_two_sum( s, e, z[1] );
}
template <>
inline void baileyAdd_renorm<2,1,2>(const double *x, const double *y, double *z){
// DWPlusFP with relative error <= 2u^2 + 5u^3
  double s, e;
  s = two_sum( x[0], y[0], e );
  e = FPadd_rn( x[1], e );
  z[0] = fast_two_sum( s, e, z[1] );
}
template<>
inline void baileyAdd_renorm<2,2,2>(double const *x, double const *y, double *z){
// AccurateDWPlusDW with relative error <= 3u^2 + 13u^3
  double sh, sl, th, tl;
  sh = two_sum( x[0], y[0], sl );
  th = two_sum( x[1], y[1], tl );
  sl = FPadd_rn( sl, th );
  sh = fast_two_sum( sh, sl, sl);
  sl = FPadd_rn( tl, sl );
  z[0] = fast_two_sum( sh, sl, z[1] );
}
#endif

