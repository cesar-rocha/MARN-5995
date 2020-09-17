# Utility functions for loading a section of several
# seabird CTD profiles and putting the data into 
# an xarray DataArray.
# Cesar B Rocha, 2020

import os
import numpy as np
import pandas as pd
import xarray as xr
from seabird.cnv import fCNV
import gsw

def LoadCTD(ctdfile,upcast=True):
    """ This loads Seabird CTD cnv data into python
        using the seabird module and splits 
        downcast and upcast. Use upcast=False for
        downcast.
        
        Returns
        ---
            - dictionary with cast position (longitude,latitude),
                pressure, temperature and salinity profiles 
                
    """      
        
    # Load Seabird CNV file
    cast = fCNV(ctdfile)

    # Get cast position
    latitude, longitude    = cast.attributes['LATITUDE'], cast.attributes['LONGITUDE']

    # split cast
    idx = np.where(np.sign(np.diff(cast['PRES'])) == -1)

    if upcast:
        pressure    = np.flip(cast['PRES'][idx[0][0]:]) 
        temperature = np.flip(cast['TEMP'][idx[0][0]:]) 
        salinity    = np.flip(cast['PSAL'][idx[0][0]:]) 
        typecast    = 'upcast'
    else:
        pressure    = cast['PRES'][:idx[0][0]] 
        temperature = cast['TEMP'][:idx[0][0]] 
        salinity    = cast['PSAL'][:idx[0][0]] 
        typecast    = 'downcast'

        
    # potential density
    sigma0        = gsw.density.sigma0(salinity,temperature)       # potential density referenced to p=0
        
    # return dictionary
    return {'latitude' : latitude,'longitude' : longitude, 
            'pressure' : pressure, 'temperature' : temperature,
            'salinity' : salinity, 'sigma0' : sigma0, 'type':typecast}

def Get_CTD_Section(datadir,stations):
    """ Get data from CTD section and put it 
            into a xarray DataArray """
    
    # Load and organize CTD profiles
    casts = np.arange(len(stations))
    lons, lats = [],[]

    for station, cast in zip(stations,casts):
    
        # Load CTD data
        ctdfile = os.path.join(datadir,'sam03_ctd_' + station + '.cnv')
        ctdcast = LoadCTD(ctdfile)
        
        # Create pandas series to be converted into a data array
        if cast == 0:
            Ts = pd.Series(data=ctdcast['temperature'],index=ctdcast['pressure'],name=cast)
            Ss = pd.Series(data=ctdcast['salinity'],index=ctdcast['pressure'],name=cast)
        else:
            Ts = pd.concat([Ts,pd.Series(data=ctdcast['temperature'],index=ctdcast['pressure'],name=cast)],axis=1)
            Ss = pd.concat([Ss,pd.Series(data=ctdcast['salinity'],index=ctdcast['pressure'],name=cast)],axis=1)   

        # Attributes to feed data array
        lons.append(ctdcast['longitude'])
        lats.append(ctdcast['latitude'])  
             
    # Dataframe to DataArray
    Txr = Ts.stack().to_xarray().rename({'level_0' : 'pressure', 'level_1' : 'cast'})
    Txr.name = 'temperature'
    Sxr = Ss.stack().to_xarray().rename({'level_0' : 'pressure', 'level_1' : 'cast'})
    Sxr.name = 'salinity'
    
    ctdsection = xr.merge([Txr,Sxr])
    
    # Calculate distance shallowest station [km]
    dists = np.cumsum(np.hstack([0,gsw.distance(lons,lats)]))/1e3 

    # Add other dimensions
    ctdsection = ctdsection.assign_coords({'latitude': xr.DataArray(lats, coords=[casts], dims="cast"),
                                       'longitude' : xr.DataArray(lons, coords=[casts], dims="cast"),
                                       'station' : xr.DataArray(stations, coords=[casts], dims="cast"),
                                       'distance' : xr.DataArray(dists, coords=[casts], dims="cast")
                                      }
                                      )   
    
    # Add physical units
    ctdsection.temperature.attrs['units'] = 'oC'
    ctdsection.salinity.attrs['units'] = 'psu'
    ctdsection.pressure.attrs['units'] = 'dbar'
    ctdsection.distance.attrs['units'] = 'km'
    
    return ctdsection
