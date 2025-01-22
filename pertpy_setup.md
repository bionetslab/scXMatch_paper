1. Create a new conda env.



2. Install pertpy.
```
pip install pertpy
```



3. Downgrade scipy.
```
pip install "scipy<1.13"
```



4. Install ete3
```
pip install ete3
```




5. In a python console, type
```
import pertpy
```
This will throw some errors related to an import made from jax.config.
Replace every occurence of 
```
from jax.config import config
```
that will throw an error by
```
from jax import config
```
until the pertpy import does not throw that error anymore.



6. Once the jax import is fixed, it will most likely throw a "TypeError: unsupported operand type(s) for |: 'type' and 'type'" due to the way, that default arguments are provided. Remove the type specifiers with the disjuction | so that only the default argument is provided.



7. After approxemately 10 repetitions of step 5 and 6, the import should work without errors.  