# ISDF Prototypes

This project is for prototyping algorithms associated with Inter Separable Density Fitting (ISDF), with the aim of
speeding up the evaluation of two-centre integrals $(ij|kl)$


## Algorithms

* [Weighted k-means clustering for finding interpolation points](src/isdf_prototyping/interpolation_points.py)


## Papers of Relevance

* K-means Clustering:
  * J. Chem. Theory Comput. 2018, 14, 1311−1320
  * Complex-Valued K-means Clustering for Interpolative Separable Density Fitting to Large-Scale Hybrid Functional 
    Ab Initio Molecular Dynamics with Plane-Wave Basis Sets


## Websites of Relevance

* Basics
  * [Basics of 2-electron integrals](http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf)

* K-means Clustering:
  * [Google tech talk](https://www.youtube.com/watch?v=NDAVDRFMh_0) on k-means algorithms


## Implementation Notes

### Weighted K-means and Finding Interpolation Points

**Issue with My Implementation**
SKLearn implementation does better than my implementation, even without using the k-means++ algorithm for better 
initial guess at the centroids. This suggests that there's an issue in my implementation
 
_**TODO:**_ Debug

SK Learn either uses greedy k-means++ (which I have not yet implemented) or if using random centroids, it runs 10 times
and selects the points that are most optimised.

_**TODO:**_ Either implement greedy k-means++ or have the random seeding algorithm run ~ 10 times and choose the best set of
centroids.

**Implement QD Decomposition as a Reference**
Should also implement QR decomposition for comparison

