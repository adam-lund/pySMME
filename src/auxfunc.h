/*
    A number of functions utilized by ....cpp.

    Copyright (C) 2022 Adam Lund

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "dwt.h"
#include "dwt3.h"
#include "dwt2.h"
#include <math.h>
using namespace std;
using namespace arma;

////////////////////////////////// Auxiliary functions
//////////////////// Direct RH-transform of a flat 3d array (matrix) M by a matrix X
arma::mat RHmat(arma::mat const& X, arma::mat const& M,int col, int sli){

int rowx = X.n_rows;

////matrix multiply
arma::mat XM = X * M;

////make matrix into rotated (!) cube (on matrix form)
arma::mat Mnew(col, sli * rowx);
for (int s = 0; s < sli; s++) {

for (int c = 0; c < col; c++) {

for (int r = 0; r < rowx; r++) {

Mnew(c, s + r * sli) = XM(r, c + s * col);

}

}

}

return Mnew;

}
///
mat cube_mult(cube C, mat M){

int G = C.n_slices, p = M.n_rows;
mat V(p, G);
for(int i = 0; i < G; i++){V.col(i) = C.slice(i) * M;}

return V;

}
///
field<mat> field_mult(field<mat> F, mat M){

int G = F.size();
field<mat> V(G, 1);
for(int i = 0; i < G; i++){V(i, 0) = F(i, 0) * M;}

return V;

}

////////////////// empirical explained variance function
arma::vec eev(arma::mat XBeta, arma::cube Z, int ng){

arma::vec eevar(Z.n_slices);
double sumXBeta2 = accu(pow(XBeta, 2));
int G = Z.n_slices;
for(int j = 0; j < G; j++){
    eevar(j) = (2 * accu(XBeta  %  Z.slice(j))  - sumXBeta2) / ng;
    }
//for(int j = 0; j < Z.n_slices; j++) {eevar(j) = (2 * XBeta  *  Z.slice(j)  - sumXBeta2) / ng;}

return eevar;

}

arma::vec eev_f(arma::field<mat> PHIX, arma::field<mat> Z, arma::vec n){

int G = Z.size();
arma::vec eevar(G);

for(int j = 0; j < G; j++){
eevar(j) = (2 * as_scalar(PHIX(j, 0).t() * Z(j, 0)) - accu(pow(PHIX(j, 0), 2))) / n(j);
}

return eevar;

}

//////////////////// softmax loss
double softmaxloss(arma::vec h, double c, bool lse){

if(lse == true){//use log sum exp

double k =  max(h);
return log(accu(exp(c * (h - k)))) / c +  k;

}else{return accu(exp(c * h));}

}
//////////////////// gradloss
arma::mat gradloss(arma::cube const& PhitZ, arma::mat const& XtXb, arma::vec const& h,
                   int ng, double c, bool lse){

arma::mat gradout(XtXb.n_rows, XtXb.n_cols);
gradout.fill(0);

if(lse == true){
double  k = max(h);
double tmp =  accu(exp(c * (h - k)));
for(int j = 0; j < PhitZ.n_slices; j++){
gradout = exp(c * (h(j) - k)) * (XtXb - PhitZ.slice(j)) + gradout;
}

return 2 * gradout / (tmp * ng);

}else{

for(int j = 0; j < PhitZ.n_slices; j++){
    gradout = exp(c * h(j)) * (XtXb - PhitZ.slice(j)) + gradout;
    }

return 2 * c * gradout / ng;

}

}

//////////////////// gradloss_f
arma::mat gradloss_f(arma::mat const& PhitZ, arma::mat const& XtXb, arma::vec const& h,
                     vec n, double c, bool lse){

arma::mat gradout(XtXb.n_rows, 1); //px1
gradout.fill(0);
int G = PhitZ.n_cols;

if(lse == true){

double  k = max(h);
double tmp =  accu(exp(c * (h - k)));
for(int j = 0; j < G; j++){
gradout = exp(c * (h(j) - k)) * (XtXb.col(j) - PhitZ.col(j)) / n(j) + gradout;
}
return 2 * gradout / tmp;

}else{

for(int j = 0; j < G; j++){
gradout = exp(c * h(j)) * (XtXb.col(j) - PhitZ.col(j)) / n(j) + gradout;
}
return 2 * c * gradout;

}

}

//////////////////// Sum of squares function
double sum_square(arma::mat const& x){return accu(x % x);}

//////////////////// Proximal operator for the l1-penalty (soft threshold)
arma::mat prox_l1(arma::mat const& zv, arma::mat const& gam){

return (zv >= gam) % (zv - gam) + (zv <= -gam) % (zv + gam);

}

//////////////////// The weighted (gam = penaltyfactor * lambda) l1-penalty function
double l1penalty(arma::mat const& gam, arma::mat const& zv){return accu(gam % abs(zv));}

//////////////////// The weighted (gam = penaltyfactor * lambda) scad-penalty function
double scadpenalty(arma::mat const& gam, double a, arma::mat const& zv){

arma::mat absbeta = abs(zv);

return accu(gam % absbeta % (absbeta <= gam)
                - (pow(zv, 2) - 2 * a * gam % absbeta + pow(gam, 2)) /
                    (2 * (a - 1)) % (gam < absbeta && absbeta <= a * gam)
                + (a + 1) * pow(gam, 2) / 2 % (absbeta > a * gam));

}

//////////////////   wavelet transform for 1d,2d,3d
arma::mat wt(arma::mat x, int dim, int L, 
             double *h,//Rcpp::NumericVector h,
             double *g,//Rcpp::NumericVector g, 
             int J, int p1, int p2, int p3, arma::mat inout){
int nx = p1, ny = p2, nz = p3, p = p1 * p2 * p3, size, endind = 0;

double* dat = nullptr; //= new double[2];

if(dim == 3){
size = nx * ny * nz / 8;
double *LLL = new double[size], *HLL = new double[size], *LHL = new double[size],
       *LLH = new double[size], *HHL = new double[size], *HLH = new double[size],
       *LHH = new double[size], *HHH = new double[size];
for(int j = 0; j < J; j++) {
if(j == 0){
three_D_dwt(x.memptr(), &nx, &ny, &nz, &L, h,g, LLL, HLL, LHL, LLH,HHL, HLH, LHH, HHH);
}else{
three_D_dwt(dat, &nx, &ny, &nz, &L, h,g, LLL, HLL, LHL, LLH, HHL,
            HLH, LHH, HHH);
}

if(j < J - 1) {
delete [] dat;
dat = new double[size];
std::copy(LLL, LLL + size, dat);
std::copy(HLL, HLL + size, inout.memptr() + endind);
std::copy(LHL, LHL + size, inout.memptr() + endind + size);
std::copy(LLH, LLH + size, inout.memptr() + endind + size * 2);
std::copy(HHL, HHL + size, inout.memptr() + endind + size * 3);
std::copy(HLH, HLH + size, inout.memptr() + endind + size * 4);
std::copy(LHH, LHH + size, inout.memptr() + endind + size * 5);
std::copy(HHH, HHH + size, inout.memptr() + endind + size * 6);
endind = endind + 7 * size;

delete [] LLL;
delete [] HLL;
delete [] LHL;
delete [] LLH;
delete [] HHL;
delete [] HLH;
delete [] LHH;
delete [] HHH;

nx = nx / 2, ny = ny / 2, nz = nz / 2;
size = nx * ny * nz / 8;
LLL = new double[size], HLL = new double[size], LHL = new double[size],
LLH = new double[size], HHL = new double[size], HLH = new double[size],
LHH = new double[size], HHH = new double[size];

}else{
std::copy(HLL, HLL + size, inout.memptr() + endind);
std::copy(LHL, LHL + size, inout.memptr() + endind + size);
std::copy(LLH, LLH + size, inout.memptr() + endind + size * 2);
std::copy(HHL, HHL + size, inout.memptr() + endind + size * 3);
std::copy(HLH, HLH + size, inout.memptr() + endind + size * 4);
std::copy(LHH, LHH + size, inout.memptr() + endind + size * 5);
std::copy(HHH, HHH + size, inout.memptr() + endind + size * 6);
std::copy(LLL, LLL + size, inout.memptr() + endind + size * 7);
endind = endind + 7 * size;

delete [] LLL;
delete [] HLL;
delete [] LHL;
delete [] LLH;
delete [] HHL;
delete [] HLH;
delete [] LHH;
delete [] HHH;
delete [] dat;

}
}

}else if(dim == 2){
  size = nx * ny  / 4;
double *LL = new double[size], *HL = new double[size], *LH = new double[size],
 *HH = new double[size];
for(int j = 0; j < J; j++) {
if(j == 0){
two_D_dwt(x.memptr(), &nx, &ny, &L, h, g, LL, LH, HL, HH);
}else{
two_D_dwt(dat, &nx, &ny, &L, h, g, LL, LH, HL, HH);
}

if(j < J - 1) {

delete [] dat;
dat = new double[size];
std::copy(LL, LL + size, dat);
std::copy(LH, LH + size, inout.memptr() + endind);
std::copy(HL, HL + size, inout.memptr() + endind + size);
std::copy(HH, HH + size, inout.memptr() + endind + size * 2);
endind = endind + 3 * size;

delete [] LL;
delete [] LH;
delete [] HL;
delete [] HH;

nx = nx / 2, ny = ny / 2;
size = nx * ny  / 4;
LL = new double[size], HL = new double[size], LH = new double[size],
HH = new double[size];

}else{

std::copy(LH, LH + size, inout.memptr() + endind);
std::copy(HL, HL + size, inout.memptr() + endind + size);
std::copy(HH, HH + size, inout.memptr() + endind + size * 2);
std::copy(LL, LL + size, inout.memptr() + endind + size * 3);
endind = endind + 3 * size;

delete [] LL;
delete [] LH;
delete [] HL;
delete [] HH;
delete [] dat;

}
}

}else{//1d
size = nx / 2;
double *V = new double[size], *W = new double[size];

for(int j = 0; j < J; j++) {

if(j == 0){
dwt(x.memptr(), &nx, &L, h, g, W, V);
}else{
dwt(dat, &nx, &L, h, g, W, V);
}

if(j < J - 1) {
delete [] dat;
dat = new double[size];
std::copy(V, V + size, dat);
std::copy(W, W + size, inout.memptr() + endind);
endind = endind + size;
delete [] V;
delete [] W;
nx = nx / 2;
size = nx / 2;
V = new double[size], W = new double[size];
}else{
std::copy(W, W + size, inout.memptr() + endind);
std::copy(V, V + size, inout.memptr() + endind + size);
delete [] V;
delete [] W;
delete [] dat;

}
}

}
return  inout;

}

////////////////// inverse wavelet transform for 1d,2d,3d
arma::mat iwt(arma::mat x, int dim, int L, double *h, double *g,
              int J, int p1, int p2, int p3 ,
              arma::mat inout){
int nx = p1 / pow(2, J), ny = p2 / pow(2, J), nz = p3 / pow(2, J);
int p = p1 * p2 * p3;
int size, startind;
double *im = nullptr;

if(dim == 3){
double *LLL  = nullptr , *HLL  = nullptr , *LHL = nullptr,*LLH  = nullptr,
  *HHL  = nullptr, *HLH  = nullptr, *LHH  = nullptr, *HHH  = nullptr;

for(int j = 0; j < J ; j++) {
size = nx * ny * nz;
LLL = new double[size], HLL = new double[size], LHL = new double[size],
LLH = new double[size], HHL = new double[size], HLH = new double[size],
LHH = new double[size], HHH = new double[size];
if(j == 0){
startind = p - 8 * size;
std::copy(x.memptr() + startind + 7 * size, x.memptr() + startind + 8 * size, LLL);
}else{
startind = startind - 7 * size;
std::copy(im, im +  size, LLL);
delete [] im;
}
//todo this copy seems inefficient...can be done with pointers?
std::copy(x.memptr() + startind, x.memptr() + startind + size, HLL);
std::copy(x.memptr() + startind + size, x.memptr() + startind + 2 * size, LHL);
std::copy(x.memptr() + startind + 2 * size, x.memptr() + startind + 3 * size, LLH);
std::copy(x.memptr() + startind + 3 * size, x.memptr() + startind + 4 * size, HHL);
std::copy(x.memptr() + startind + 4 * size, x.memptr() + startind + 5 * size, HLH);
std::copy(x.memptr() + startind + 5 * size, x.memptr() + startind + 6 * size, LHH);
std::copy(x.memptr() + startind + 6 * size, x.memptr() + startind + 7 * size, HHH);
im = new double[8 * size];

three_D_idwt(LLL, HLL, LHL, LLH, HHL, HLH, LHH, HHH, &nx, &ny, &nz, &L, h,g , im);
nx = nx * 2, ny = ny * 2, nz = nz * 2;
delete [] LLL;
delete [] HLL;
delete [] LHL;
delete [] LLH;
delete [] HHL;
delete [] HLH;
delete [] LHH;
delete [] HHH;

}

}else if(dim == 2){//2d

double *LL  = nullptr , *HL  = nullptr , *LH = nullptr, *HH  = nullptr;

for(int j = 0; j < J ; j++) {
size = nx * ny;
LL = new double[size], HL = new double[size], LH = new double[size],
HH = new double[size];
    if(j == 0){
      startind = p - 4 * size;
      std::copy(x.memptr() + startind + 3 * size, x.memptr() + startind + 4 * size, LL);
    }else{
      startind = startind - 3 * size;
      std::copy(im, im +  size, LL);
      delete [] im;
    }
    //todo this copy seems inefficient...can be done with pointers?
    std::copy(x.memptr() + startind, x.memptr() + startind + size, LH);
    std::copy(x.memptr() + startind + size, x.memptr() + startind + 2 * size, HL);
    std::copy(x.memptr() + startind + 2 * size, x.memptr() + startind + 3 * size, HH);
    im = new double[4 * size];
    two_D_idwt(LL, LH, HL, HH, &nx, &ny, &L, h, g, im);
    nx = nx * 2, ny = ny * 2;
    delete [] LL;
    delete [] HL;
    delete [] LH;
    delete [] HH;

  }


}else{//1d
double *V = nullptr, *W  = nullptr;

for(int j = 0; j < J ; j++) {
size = nx ;
V = new double[size], W = new double[size] ;
if(j == 0){
startind = p - 2 * size;
std::copy(x.memptr() + startind + size, x.memptr() + startind + 2 * size , V);
}else{
startind = startind - size;
std::copy(im, im + size, V);
}
//todo this copy seems inefficient...can be done with pointers?
std::copy(x.memptr() + startind, x.memptr() + startind + size , W);
delete [] im;
im = new double[2 * size];
idwt(W, V, &nx,   &L, h,  g, im);
nx = nx * 2 ;
delete [] V;
delete [] W;

}
}

std::copy(im, im + p1 * p2 * p3, inout.memptr());
delete [] im;
return  inout;

}



