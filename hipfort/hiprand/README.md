# Computing `pi`

Hipfort provides interfaces for various highly optimized library. In this exercise `hip_rand` library is used to accelerate the computation of `pi` using Monte Carlo method. In this method **random** `(x, y)` points are generated in a 2-D plane with domain as a square of side *2r* units centered on `(0,0)`. 

The folder  [hiprand_example](hiprand_example/) shows how to call the `hiprand` for generating single precision uniform random distributed nubmbers for calculation the value of `pi`. A circle of radius **r** centered at `(0,0)` will fit perfectly inside. The ratio between the area of circle and the square is `pi/4`. If enough numbers are uniformely distrbuted numbers are generate one can assume that number of poits generated in the square or the inside circle are direct proportionally to the areas. In order to get the value of `pi/4` one just needs to take the ratio between the number of points which ar inside the circle and the total number of points generated insidde the square.

<figure>
  <img src="img/pi_MC.png" width="50%" alt="Pi Monte Carlo">
  <figcaption> </figcaption>
</figure>


For more examples of HIPFORT  check also the [HIPFORT repository](https://github.com/ROCmSoftwarePlatform/hipfort/tree/develop/test).
