function Y = logpdf_gmm(X, Ws, MUs, COVs)
%LOGPDF_GMM Log density function for multivariate Gaussian mixture distribution.
%   Y = LOGPDF_GMM(X,Ws,MUs,COVs) returns a row vector of values of the log
%   probability density function for multivariate Gaussian mixture
%   distribution with mixture component mean vectors given by columns of
%   D-by-M matrix MUs, covariance matrices given by D-by-D-by-M matrix COVs
%   and vector of weights Ws. Y are values of the distribution evaluated at
%   the vectors given by collumns of X, where D is dimensionality of data
%   and M is number of mixture components.
%
%   Alternatively, COVs can be D-by-M matrix, in wich case its columns
%   represent diagonal of diagonal covariance matrices.

M  = size(MUs, 2);
Y  = -inf(1, size(X, 2)); 

if ndims(COVs) == ndims(MUs) && all(size(COVs) == size(MUs))
  for ii = 1:M
    Y = logsumexp([Y;  log(Ws(ii)) + logpdf_gauss(X, MUs(:,ii), COVs(:,ii))]);
  end
else
  for ii = 1:M
    Y = logsumexp([Y;  log(Ws(ii)) + logpdf_gauss(X, MUs(:,ii), COVs(:,:,ii))]); 
  end
end
