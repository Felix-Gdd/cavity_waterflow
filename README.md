# Pre and post processing python tools to model water flow in sub-glacial cavities
## Pre-proc : (functions.py and main.ipynb)
input data == a 2D (x,y) structured grid containing z-values of the bottom and top of the cavities. 
- Filtering to get clean and retain only percolating cavities trough the domain.  
Output == 'wind-tunnel' 3D mesh used to carve out a volummetric mesh via OpenFOAM tools (blockMesh and snappyHexMesh) 

## Post-proc : 
input data == .vtu output files of the simpleFom simulation, flowrate.dat 
- Re griding of the values on a 3D structured grid
- z-averaged xy plots
- computes the average SS flowrates 
