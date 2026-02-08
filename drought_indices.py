# Imports
import os, sys
import re
import gc
import numpy as np
import pickle
from scipy import stats
from scipy.special import gamma, gammainc
from datetime import datetime, timedelta
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from matplotlib import colorbar as mcolorbar


#%%
#################################################################
def calculate_spi(precip, time, savedata = True, dir = '../Data/drought', period = 'last1000', compress = False):
    '''
    Calculate the standardized precipitation index (SPI) from precipitation data. 
    SPI index is on the same time scale as the input data.
    
    Inputs:
    :param precip: Input precipitation data (in kg m^-2 s^-1; should be for over 10+ years). Time x lat x lon format
    :param time: Vector of datetimes corresponding to the timestamp in each timestep in precip.
    :param savedata: Parameter to save the data in pickle file or not
    :param dir: Directory to write the data to
    :param period: The period the data covers. Used to name the data file.
    :param compress: Boolean of whether to compress the SPI to float32 to save space. Note this makes the data half as precise.
    
    Outputs:
    :param spi: The SPI drought index.
    '''
    
    T, I, J = precip.shape
    
    precipitation = precip*3600*24 # Convert precipitation from kg m^-2 s^-1 to mm. Standard precip should be >> 1e-5 now.
    
    # Individual months and days are needed for later comparisons
    years = np.array([date.year for date in time])
    months = np.array([date.month for date in time])
    days = np.array([date.day for date in time])
    
    N_per_year = int(T/len(np.unique(years))) # Number of observations per year
    N = int(len(np.unique(years))) # Number of observations = number of years (calculation is performed for each Jan. in the dataset, each Feb., etc. separately
    
    # Initialize the spi values
    spi = np.ones((T, I ,J)) * np.nan
    
    # Reshape everything into 2D arrays for the easier calculations (fewer embedded loops).
    precipitation = precipitation.reshape(T, I*J)
    #alpha = alpha.reshape(I*J)
    #beta = beta.reshape(I*J)
    spi = spi.reshape(T, I*J)
    
    for t, date in enumerate(time[:N_per_year]):
        ind = np.where( (months == date.month) & (days == date.day) )[0]
        print(date)
        # Precipitation is dsitribution according to a gamma distribution. Find the parameters of the gamme distribution
        A = np.log(np.nanmean(precipitation[ind,:]+1e-4, axis = 0)) - np.nansum(np.log(precipitation[ind,:]+1e-4), axis = 0)/N
        alpha = (1/(4*A)) * (1 + np.sqrt(1+4*A/3))
        beta  = np.nanmean(precipitation[ind,:], axis = 0)/alpha
    
        for ij in range(I*J):
            #if (ij%1000) == 0:
                #print('%d/%d'%(int(ij/1000), int(I*J/1000)))
            # Transform the values into a normal distribution with mean 0 and standard deviation 1 using the inverse CDF method.
            # That is, if P has a gamma distribution, then cdf(P) is a uniform distribution. Then cdf_n^-1(cdf(P)) is normally distributed.
        
            # cdf_n^-1 is the inverse of the normal cdf
            # the ppf, percent point function, is the inverse cdf. Its default arguements are mean (loc) = 0, and std (scale) = 1

            cdf = gammainc(alpha[ij], precipitation[ind,ij]/beta[ij])
        
            q = len(np.where(precipitation[ind,ij] < 0.01)[0])/N # Get the weight of the number of points with precipitation at (or very close to) 0
            cdf = q + (1-q)*cdf # Ajdust the cdf to account for the precip = 0 grid points
        
            # Points where the entire cdf is 0s and 1 is biasing the data. Hard set the SPI NaN to remove the bias.
            #### This might be a class imbalance: 
            #.   Locations where no precip heavily outnumbers precip, the cdfs give no precip a probability of 1 and 0 to everything else.
            #.   This gives errors for pdf (prob 0 corresponds to -inf on the pdf, and +inf when prob is 1, or unrealistically small/large values when padding
            #.   is applied.
            #### Come back later to fix this class imbalance
            if len(np.where( (cdf > 0.999) | (cdf < 0.001) )) == T:
                spi[ind,ij] = np.nan
                continue
            
            # "Pad" the extreme cdf probability values (near 1 or 0) to prevent the calculated distribution from returning +/- inf
            cdf = np.where((cdf > 0.999), cdf-0.001, cdf)
            cdf = np.where((cdf < 0.001), cdf+0.001, cdf)
            
            spi[ind,ij] = stats.norm.ppf(cdf, loc = 0, scale = 1)

    
    # Return SPI to a 3D format
    spi = spi.reshape(T, I, J)
    
    # Compress the data?
    if compress:
        spi = spi.astype(np.float32)
    
    # Write the data?
    if savedata:
        filename = 'spi_%s.pkl'%(period)
        with open('%s/%s'%(dir, filename), 'wb') as fp:
            pickle.dump(spi, fp)
    
    return spi
   
    
#%%
##############################################

# Create functions to calculate SPEI and SAPEI
# Details for SPEI can be found in the Vicente-Serrano et al. 2010 paper.
# Details for SAPEI can be found in the Li et al. 2020b paper.

def transform_pearson3(data, time, climo = None, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Transform 3D gridded data in Pearson Type III distribution to a standard normal distribution.
    
    Inputs:
    :param data: Pearson Type III distributed data to be transformed
    :param time: Vector of datetimes corresponding to the timestamp in each timestep in precip and pet
    :param climo: Subset of data consisting of the data used to determine the parameters of Pearson Type III distribution. Default is data
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param data_norm: data in a standard normal distribution, same shape and size as data
    '''
    
    print('Initializing some values')
    
    # Climo dataset specified?
    if climo.all() == None:
        climo = data

    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in time])
        
    if months == None:
        months = np.array([date.month for date in time])
        
    if days == None:
        days = np.array([date.day for date in time])
    
    
    # Initialize some needed variables.
    T, I, J = data.shape
    T_climo = climo.shape[0]

    climo_index = np.where((years >= start_year) & (years <= end_year))[0]
    
    N = int(T/len(np.unique(years))) # Number of observations per year
    N_obs = int(T_climo/N) # Number of observations per time series; number of years in climo
    #N_obs = len(np.unique(years)) # Number of observations per time series
    
    # Define the constants given in Vicente-Serrano et al. 2010
    C0 = 2.515517
    C1 = 0.802853
    C2 = 0.010328

    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    
    frequencies = np.ones((T, I, J)) * np.nan
    PWM0 = np.ones((N, I, J)) * np.nan # Probability weighted moment of 0
    PWM1 = np.ones((N, I, J)) * np.nan # Probability weighted moment of 1
    PWM2 = np.ones((N, I, J)) * np.nan # Probability weighted moment of 2
    
    # Determine the frequency estimator and moments according to equation in section 3 of the Vicente-Serrano et al. 2010 paper
    print('Calculating moments')
    for t, date in enumerate(time[:N]):
        ind = np.where( (months[climo_index] == date.month) & (days[climo_index] == date.day) )[0]

        # Get the frequency estimator
        frequencies[ind,:,:] = (stats.mstats.rankdata(climo[ind,:,:], axis = 0) - 0.35)/N_obs

        # Get the moments
        PWM0[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**0)*climo[ind,:,:], axis = 0)/N_obs
        PWM1[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**1)*climo[ind,:,:], axis = 0)/N_obs
        PWM2[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**2)*climo[ind,:,:], axis = 0)/N_obs

    # Calculate the parameters of log-logistic distribution, using the equations in the Vicente-Serrano et al. 2010 paper
    print('Calculating Pearson Type III distribution parameters')
    beta  = (2*PWM1 - PWM0)/(6*PWM1 - PWM0 - 6*PWM2) # Scale parameter
    
    alpha = (PWM0 - 2*PWM1)*beta/(gamma(1+1/beta)*gamma(1-1/beta)) # Shape parameter
    
    gamm  = PWM0 - alpha*gamma(1+1/beta)*gamma(1-1/beta) # Origin parameter; note gamm refers to the gamma parameter

    # Obtain the cumulative distribution of the moisture deficit.
    print('Calculating the cumulative distribution of P - PET')
    F = np.ones((T, I, J)) * np.nan

    for n, date in enumerate(time[:N]):
        ind = np.where( (date.month == months) & (date.day == days) )[0]

        for t in ind:
            F[t,:,:] = (1 + (alpha[n,:,:]/(data[t,:,:] - gamm[n,:,:]))**beta[n,:,:])**-1

    # Some variables are no longer needed. Remove them to conserve memory.
    del frequencies, PWM0, PWM1, PWM2, beta, alpha, gamm
    gc.collect() # Clears deleted variables from memory

    # Finally, use this to obtain the probabilities and convert the data to a standardized normal distribution
    prob = 1 - F
    prob = np.where(prob == 0, 1e-5, prob) # Remove probabilities of 0
    prob = np.where(prob == 1, 1-1e-5, prob) # Remove probabilities of 1
    
    data_norm = np.ones((T, I, J)) * np.nan 

    # Reshape arrays into 2D for calculations
    prob = prob.reshape(T, I*J)
    data_norm = data_norm.reshape(T, I*J)
    
    # Calculate SPEI based on the inverse normal approximation given in Vicente-Serrano et al. 2010, Sec. 3
    print('Converting P - PET probabilities into a normal distribution')
    data_sign = 1
    for ij in range(I*J):
        if (ij%1000) == 0:
            print('%d/%d'%(int(ij/1000), int(I*J/1000)))
        for t in range(T):
            
            # Determine if to multiple the equation by 1 or -1, and whether to use prob or 1 - prob
            if prob[t,ij] <= 0.5:
                prob[t,ij] = prob[t,ij]
                data_sign = 1
            else:
                prob[t,ij] = 1 - prob[t,ij]
                data_sign = -1

            # Determine W
            W = np.sqrt(-2 * np.log(prob[t,ij]))

            # Calculate the normal distribution
            data_norm[t,ij] = data_sign * (W - (C0 + C1 * W + C2 * (W**2))/(1 + d1 * W + d2 * (W**2) + d3 * (W**3)))
    
    # Reshape SPEI back into a 3D array
    data_norm = data_norm.reshape(T, I, J)
    
    print('Done')
    return data_norm

def calculate_spei(precip, pet, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the standardized precipitation evaporation index (SPEI) from precipitation and potential evaporation data.
    SPEI index is on the same time scale as the input data.
    
    Full details on SPEI can be found in Vicente-Serrano et al. 2010: https://doi.org/10.1175/2009JCLI2909.1
    
    Inputs:
    :param precip: Input precipitation data (should be for over 10+ years). Time x lat x lon format
    :param pet: Input potential evaporation data (should be over 10+ years). Time x lat x lon format. Should be in the same units as precip
    :param dates: Vector of datetimes corresponding to the timestamp in each timestep in precip and pet
    :param mask: Land-sea mask for the precip and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates

    Outputs:
    :param spei: The SPEI drought index, has the same shape and size as precip/pet
    '''
    
    # Determine the moisture deficit
    D = (precip) - pet
    
    # Collect the climatological data for the ED
    D_climo = collect_climatology(D, dates, start_year = start_year, end_year = end_year)
    
    # Transform D from a Pearson Type III distribution to a standard normal distribution
    spei = transform_pearson3(D, dates, climo = D_climo, start_year = start_year, end_year = end_year, years = years, months = months, days = days)
    
    # Remove any sea data points
    # May comment this out if there is no mask
    spei = apply_mask(spei, mask)
    # spei[mask[:,:] == 0] = np.nan
    
    return spei
    
#%%
##############################################
# Create a function to collect climatological data

def collect_climatology(X, dates, start_year, end_year):
    '''
    Extract data between the beginning and ending years.
    
    Inputs:
    :param X: Input 3D data, in time x lat x lon format
    :param dates: Array of datetimes corresponding to the time axis of X
    :param start_year: The beginning year in the interval being searched
    :param end_year: The ending year in the interval being searched
    
    Outputs:
    :param X_climo: X between the specified start and end dates
    '''
    
    # Turn the start and end years into datetimes
    begin_date = datetime(start_year, 1, 1)
    end_date   = datetime(end_year, 12, 31)
    
    # Determine all the points between the start and end points
    ind = np.where( (dates >= begin_date) & (dates <= end_date) )[0]
    
    if len(X.shape) < 3:
        X_climo = X[ind]
    else:
        X_climo = X[ind,:,:]
    
    return X_climo


#%% 
##############################################
# Calculate the climatological means and standard deviations
  
def calculate_climatology(X, pentad = True):
    '''
    Calculates the climatological mean and standard deviation of gridded data.
    Climatological data is calculated for all grid points and for all timestamps in the year.
    
    Inputs:
    :param X: 3D variable whose mean and standard deviation will be calculated.
    :param pentad: Boolean value giving if the time scale of X is 
                   pentad (5 day average) or daily.
              
    Outputs:
    :param clim_mean: Calculated mean of X for each day/pentad and grid point. 
                      clim_mean as the same spatial dimensions as X and 365 (73)
                      temporal dimension for daily (pentad) data.
    :param clim_std: Calculated standard deviation for each day/pentad and grid point.
                     clim_std as the same spatial dimensions as X and 365 (73)
                     temporal dimension for daily (pentad) data.
    '''
    
    # Obtain the dimensions of the variable
    if len(X.shape) < 3:
        T = X.size
    else:
        T, I, J = X.shape
    
    # Count the number of years
    if pentad is True:
        year_len = int(365/5)
    else:
        year_len = int(365)
        
    N_years = int(np.ceil(T/year_len))
    
    # Create a variable for each day, assumed starting at Jan 1 and no
    #   leap years (i.e., each year is only 365 days each)
    day = np.ones((T)) * np.nan
    
    n = 0
    for i in range(1, N_years+1):
        if i >= N_years:
            day[n:T+1] = np.arange(1, len(day[n:T+1])+1)
        else:
            day[n:n+year_len] = np.arange(1, year_len+1)
        
        n = n + year_len
    
    # Initialize the climatological mean and standard deviation variables
    if len(X.shape) < 3:
        clim_mean = np.ones((year_len)) * np.nan
        clim_std  = np.ones((year_len)) * np.nan
    else:
        clim_mean = np.ones((year_len, I, J)) * np.nan
        clim_std  = np.ones((year_len, I, J)) * np.nan
    
    # Calculate the mean and standard deviation for each day and at each grid
    #   point
    for i in range(1, year_len+1):
        ind = np.where(i == day)[0]
        
        if len(X.shape) < 3:
            clim_mean[i-1] = np.nanmean(X[ind], axis = 0)
            clim_std[i-1]  = np.nanstd(X[ind], axis = 0)
        else:
            clim_mean[i-1,:,:] = np.nanmean(X[ind,:,:], axis = 0)
            clim_std[i-1,:,:]  = np.nanstd(X[ind,:,:], axis = 0)
    
    return clim_mean, clim_std

#%%
##############################################

# Function to remove sea data points
def apply_mask(data, mask):
    '''
    Turn sea points into NaNs based on a land-sea mask where 0 is sea and 1 is land
    
    Inputs:
    :param data: Data to be masked
    :param mask: Land-sea mask. Must have the same spatial dimensions as data
    
    Outputs:
    :param data_mask: Data with all labeled sea grids at NaN
    '''
    
    T, I, J = data.shape
    
    data_mask = np.ones((T, I, J)) * np.nan
    for t in range(T):
        data_mask[t,:,:] = np.where(mask == 1, data[t,:,:], np.nan)
        
    return data_mask
    
    
#%%
##############################################
# Create a function to import the processed .nc files

def load_nc(SName, filename, model = 'livneh',  path = './Data/Processed_Data/'):
    '''
    Load a .nc file.
    
    Inputs:
    :param SName: The short name of the variable being loaded. I.e., the name used
                  to call the variable in the .nc file.
    :param filename: The name of the .nc file.
    :param model: String indicating which model is being loaded
    :param sm: Boolean determining if soil moisture is being loaded (an extra variable and dimension, level,
               needs to be loaded).
    :param path: The path from the current directory to the directory the .nc file is in.
    :param narr: Boolean indicating whether NARR data is being loaded (longitude values have to be corrected if so)
    
    Outputs:
    :param X: A dictionary containing all the data loaded from the .nc file. The 
              entry 'lat' contains latitude (space dimensions), 'lon' contains longitude
             (space dimensions), 'date' contains the dates in a datetime variable
             (time dimension), 'month' 'day' are the numerical month
             and day value for the given time (time dimension), 'ymd' contains full
             datetime values, and 'SName' contains the variable (time x lat x lon).
    '''
    
    # Initialize the directory to contain the data
    X = {}
    DateFormat = '%Y-%m-%d %H:%M:%S'
    
    with Dataset(path + filename, 'r') as nc:
        # Load the grid
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]

        X['lat'] = lat
        X['lon'] = lon
        
        # Collect the time information
        time = nc.variables['time'][:]
        #X['time'] = time
        # Start year is 1915 for Livneh, 1900 for PRISM
        if model == 'livneh':
            start_year = 1915
        elif model == 'prism':
            start_year = 1900
        dates = np.asarray([datetime (start_year,1,1) + timedelta(days=ndays) for ndays in time])
        
        X['date'] = dates
        X['year']  = np.asarray([d.year for d in dates])
        X['month'] = np.asarray([d.month for d in dates])
        X['day']   = np.asarray([d.day for d in dates])
        X['ymd']   = np.asarray([datetime(d.year, d.month, d.day) for d in dates])

        # Collect the data itself
        # Assumes 3D data, in the shape of time by x by y
        X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X

def write_nc(data, lat, lon, dates, filename = 'tmp.nc', var_sname = 'tmp', path = './'):
    '''
    Add documentation
    '''
    
    T, I, J = data.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        
        # Create the dimeneions
        nc.createDimension('lat', size = I)
        nc.createDimension('lon', size = J)
        nc.createDimension('time', size = T)
        
        # Create the variables
        nc.createVariable('lat', lat.dtype, ('lat'))
        nc.createVariable('lon', lon.dtype, ('lon'))
        nc.createVariable('time', str, ('time'))
        nc.createVariable(str(var_sname), data.dtype, ('time', 'lat', 'lon'))
        
        # Dump the variables
        nc.variables['lat'][:] = lat[:]
        nc.variables['lon'][:] = lon[:]
        nc.variables['time'][:] = np.asarray([date.strftime('%Y-%m-%d') for date in dates])
        nc.variables[str(var_sname)][:,:,:] = data[:,:,:]


def do_map(var, lat, lon,title="SPI_1950", savename="SPI_1950.png"):
    cmin= -2; cmax=2; cint=0.2
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs =len (clevs)
    cname="BrBG"
    cmap= plt.get_cmap(name =cname,lut=nlevs)
    lat_int = 10
    lon_int = 20
    LatLabel = np.arange(-90,90, lat_int)
    LonLabel = np.arange(-180,180, lon_int)

    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()

    fig_proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=[12,10])
    ax= fig.add_subplot(1,1,1, projection=fig_proj)
    ax.set_title(title)
    ax.add_feature(cfeature.STATES)
    ax.coastlines(edgecolor="black")
    cs = ax.pcolormesh(lon,lat, var, vmin = cmin, vmax=cmax,
                       cmap=cmap, transform=fig_proj)
    #ax.set_extent([-130, -65, 23.5, 48.5])
    #ax.set_extent([-105, -90.5, 25, 39])
    print([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)])
    ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)])
    cbax =fig.add_axes([0.925, 0.30, 0.020, 0.40])
    cbar=fig.colorbar(cs, cax=cbax, orientation="vertical")
    cbar.ax.set_ylabel("SPI")
    plt.savefig(savename)
    plt.show(block=False)
    
if __name__ == "__main__":
    #load the data
    prec=load_nc("prec", "prec.1950.nc", "/data/deluge/reanalysis/REANALYSIS/Livneh/")
    spi=calculate_spi((prec["prec"]),savedata=False)
    mean_spi= np.nanmean(spi, axis=0)
    lon, lat = np.meshgrid(prec["lon"], prec["lat"])
    do_map(mean_spi, lat, lon)

