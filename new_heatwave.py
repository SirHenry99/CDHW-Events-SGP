# Load libraries
import os, sys
import re
import gc
from scipy import interpolate
from scipy import signal
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from matplotlib import colorbar as mcolorbar

# Load custom functions
from drought_indices import load_nc
from heatwave import *

######
# Function to load in an merge datasets
def load_temp_data(region, time_scale, model = 'livneh', data_type = 'mean', summer_months = True, path = './'):
    '''
    Load in all the temperature data for 1950 - 2007 for a designated subregion of the U.S.
    
    Inputs:
    :param region: String. The region (as an accronym) the data will be subsetted to. Accepted entries: SGP, NGP, SW, NW, SE, NE
    :param time_scale: String. The time scale the data will be output as. Accepted entries: daily, monthly
    :param model: String. Reanalysis model being loaded in
    :param data_type: String indicating whether the mean, maximum, or minimum temperature values are loaded; accepted entries are 'mean', 'max', or 'min'
    :param summer_months: Boolean. Indicate whether to load in only the summer months (May - October) or not
    :param path: String. The path leading to the dataset
    
    Outputs:
    :param prec_set: Precipitation data for 1950 - 2007 for the given region. Data is time x lat x lon format
    :param lat: 1D array of floats. Latitude values for prec_set/region
    :param lon: 1D array of floats. Longitude values for prec_set/region
    :param dates: 1D array of datetimes. Timestamps for each temporal entry in prec_set
    '''

    if region == 'SGP':
        lower_lat = 25
        upper_lat = 39
        lower_lon = 255
        upper_lon = 269.5
    elif region == 'NGP':
        lower_lat = 39
        upper_lat = 50
        lower_lon = 255
        upper_lon = 269.5
    elif region == 'SW':
        lower_lat = 25
        upper_lat = 41
        lower_lon = 235
        upper_lon = 255
    elif region == 'NW':
        lower_lat = 41
        upper_lat = 50
        lower_lon = 235
        upper_lon = 255
    elif region == 'SE':
        lower_lat = 25
        upper_lat = 38
        lower_lon = 269.5
        upper_lon = 292
    elif region == 'NE':
        lower_lat = 38
        upper_lat = 50
        lower_lon = 269.5
        upper_lon = 292

    # PRISM longitude goes from -180 to 180
    if model == 'prism':
        lower_lon = lower_lon - 360
        upper_lon = upper_lon - 360
        
    if data_type.lower() == 'mean':
        fname_base = 'surfT'
        T_key = 't2m'
    elif data_type.lower() == 'max':
        fname_base = 'prismmaxtemp'
        T_key = 'tmax'
    elif data_type.lower() == 'min':
        fname_base = 'prismmintemp'
        T_key = 'tmin'
    else:
        assert("Only accepted entries for data_type is 'mean', 'max', or 'min'!")
        
        
    # Load in an example of the data to collect grid size, and the number of days in the summer months of interest
    tair=load_nc(T_key, "%s.2018.nc"%fname_base, model = model, path = path)
    region_lat_ind = np.where((tair['lat'] >= lower_lat) & (tair['lat'] <= upper_lat))[0]
    region_lon_ind = np.where((tair['lon'] >= lower_lon) & (tair['lon'] <= upper_lon))[0]
    tair[T_key] = tair[T_key][:,region_lat_ind,:]
    tair[T_key] = tair[T_key][:,:,region_lon_ind]
    T, I, J = tair[T_key].shape
    
    # Collect the years to load
    if model == 'livneh':
        years=np.arange(1950, 2007)
    elif model == 'prism':
        years = np.arange(1981, 2023)

    # Determine the size of the time axis
    if summer_months:
        ind_summer = np.where((tair['month'] >= 5) & (tair['month'] <= 10))[0]
    
        # Months of interest
        months = np.arange(5,10+1)
    
        # Number of days in the months of interest
        ndays = len(ind_summer)
        
    else:
        # Months of interest
        months = np.arange(1,12+1)
    
        # Number of days in the months of interest
        ndays = 365
    
    # Construct the filenames
    filenames=["%s.%s.nc"%(fname_base, year) for year in years]

    # Initialize the data
    if time_scale == 'daily':
        tair_set=np.ones((len(years)*ndays, I, J))*np.nan
        
    elif time_scale == 'monthly':
        tair_set=np.ones((len(years)*len(months), I, J))*np.nan

    # Load in each file, subset them, and load them into the overall dataset
    t=0
    for filename in filenames:
        # Output progress
        print("working on %s" %filename)
    
        # Load the data
        tair=load_nc(T_key, filename, model = model, path = path)
    
        # Subset to the region
        region_lat_ind = np.where((tair['lat'] >= lower_lat) & (tair['lat'] <= upper_lat))[0]
        region_lon_ind = np.where((tair['lon'] >= lower_lon) & (tair['lon'] <= upper_lon))[0]
    
        tair[T_key] = tair[T_key][:,region_lat_ind,:]
        tair[T_key] = tair[T_key][:,:,region_lon_ind]
        tair['lat'] = tair['lat'][region_lat_ind]
        tair['lon'] = tair['lon'][region_lon_ind]
    
        # Subset to the summer months.....
        if summer_months:
            ind_summer = np.where((tair['month'] >= 5) & (tair['month'] <= 10))[0]

            tair[T_key] = tair[T_key][ind_summer,:,:]
            tair['month'] = tair['month'][ind_summer]
            tair['date'] = tair['date'][ind_summer]
        else:
            # Check for and remove leap days
            ind = np.where((tair['month'] == 2) & (tair['day'] == 29))[0]
            
            tair[T_key] = np.delete(tair[T_key], ind, axis = 0)
            tair['month'] = np.delete(tair['month'], ind)
            tair['date'] = np.delete(tair['date'], ind)
            #print(tair[T_key].shape)
    
        # Correection for PRISM in 2021
        #if (model == 'prism') & (filename == 'prec.2021.nc'):
        #    prec['prec'][prec['prec'] >= 273.15] = prec['prec'][prec['prec'] >= 273.15] - 273.15
        
        # Add the data to the merged dataset
        if time_scale == 'daily':
            tair_set[t:t+ndays,:,:] = tair[T_key]
    
            t = t + ndays
            
        elif time_scale == 'monthly':
            for month in months:
                # Average the data to the monthly scale and add to the merged dataset
                ind= np.where(month==tair['month'])[0]
                tair_set[t,:,:]=np.nanmean(tair[T_key][ind,:,:], axis=0)
                t=t+1

    
    # This process will remove the mask on the data, replacing them with the default 1e20 values. Turn these masked values to NaNs.
    tair_set[tair_set >= 1e10] = np.nan
    
    # Construct the time stamps for the data
    dates = []
    if time_scale == 'daily':
        for year in years:
            for t in range(ndays):
                dates.append(datetime.datetime(year, months[0], 1) + datetime.timedelta(days = t))
    
    elif time_scale == 'monthly':
        for year in years:
            for month in months:
                dates.append(datetime.datetime(year,month,1))
    
    # Turn the timestamps into an array    
    dates=np.array(dates)
    
    #if model == 'prism':
    #    prec_set[prec_set > 150] = np.nan
    
    lat = tair['lat']
    lon = tair['lon']
    
    return tair_set, lat, lon, dates

def do_map(var, lat, lon,title="temp_1950", var_name = 'temp', savename="temp_1950.png"):
    #cmin= 13; cmax=38; cint=0.75 #25..40
    cmin= 0; cmax=12; cint=1.0
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs =len (clevs)
    cname="Reds"
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
    cbar.ax.set_ylabel(var_name)
    plt.savefig(savename)
    plt.show(block=False)


###########
# Function to standardize a variable
def standardize_values(var, dates, ndays):
    '''
    tmp. BE SURE TO FILL THIS IN
    '''
    
    
    # Make the months, and days variables
    months = np.array([date.month for date in dates])
        
    days = np.array([date.day for date in dates])
    
    # Contrust an array of datetimes for 1 year   
    dates_one_year=dates[:ndays]
    
    # Get the size of the data
    T, I, J = var.shape
    
    # Initialize the array of the standardized variable
    var_standard = np.ones((T, I, J)) * np.nan
    
    # Standardize the variable for each grid point according to the time of year
    for t, day in enumerate(dates_one_year):
        ind = np.where((months == day.month) & (days == day.day))[0]
        #var_standard[ind,:,:] = (var[ind,:,:] - np.nanmean(var[ind,:,:], axis = 0))/np.nanstd(var[ind,:,:], axis = 0)
        var_standard[ind,:,:] = (var[ind,:,:] - np.nanmean(var[ind,:,:], axis = 0))
    return var_standard


# Begin body of script
if __name__ == '__main__':
    # Load in mean temperature data
    print('Loading data...')
    region = 'SGP'
    tair = {}
    #tair['tmean'], tair['lat'], tair['lon'], tair['dates'] = load_temp_data(region, 'daily', model = 'prism', data_type = 'mean', summer_months = False, path = './')
    #tair['tmean'] = tair['tmean'].astype(np.float32)
    tair['tmax'],tair['lat'], tair['lon'], tair['dates'] = load_temp_data(region, 'daily', model = 'prism', data_type = 'max', summer_months = False, path = './prismmaxtemp/')
    tair['tmax'] = tair['tmax'].astype(np.float32)
    print('Data size: ', tair['tmax'].shape)
    
    # T, I, J = tair['tmean'].shape
    # print('Data size: ', tair['tmean'].shape)
    T, I, J = tair['tmax'].shape
    print('Data size: ', tair['tmax'].shape)
    
    # Determine the number of days in each year
    years = np.array([date.year for date in tair['dates']])
    ndays = int(tair['dates'].size/len(np.unique(years)))
    
    start_year = years[0]
    
    # Also make months and days variables
    months = np.array([date.month for date in tair['dates']])
    days = np.array([date.day for date in tair['dates']])
    
    # Standardize the temperature data
    # print('Standardizing temperature...')
    # tair['tmean'] = standardize_values(tair['tmean'], tair['dates'], ndays) ######################
    
    # Determine the spatial mean of the anomalies
    print('Calculating spatial mean...')
    # tair['tmean_ts'] = np.nanmean(tair['tmean'].reshape(T, I*J, order = 'F'), axis = -1)
    tair['tmax_ts'] = np.nanmean(tair['tmax'].reshape(T, I*J, order = 'F'), axis = -1)
    
    # Calculate the percentile for each point in the time series based on the time of year
    one_year = tair['dates'][:ndays]
    
    percentiles = [90]
    min_days = [5]
    for percentile in percentiles:
        print('Calculating percentiles...')
        ta_percentiles = np.ones((one_year.size)) * np.nan
        for t, day in enumerate(one_year):
            ind = np.where((months == day.month) & (days == day.day))[0]
            # ta_percentiles[t] = np.nanpercentile(tair['tmean_ts'][ind], percentile)
            ta_percentiles[t] = np.nanpercentile(tair['tmax_ts'][ind], percentile)
            
        # Determine the heat wave initial start and end dates based on percentiles
        print('Determining initial HW dates...')
        hw_initaldates(ta_percentiles, tair['tmax_ts'], './', start_year, region, percentile) ##########
        
        
#     # Load the maximum temperature data
#     print('Loading maximum temperature data...')
#     tair['tmax'], _, _, _ = load_temp_data(region, 'daily', model = 'prism', data_type = 'max', summer_months = False, path = './')
#     tair['tmax'] = tair['tair'].astype(np.float32)
#     print('Data size: ', tair['tmax'].shape)
    
#     # Load the minimum temperature data
#     print('Loading minimum temperature data...')
#     tair['tmin'], _, _, _ = load_temp_data(region, 'daily', model = 'prism', data_type = 'min', summer_months = False, path = './')
#     tair['tmin'] = tair['tmin'].astype(np.float32)
#     print('Data size: ', tair['tmin'].shape)
    
#     # Standardize min and maximum temperatures
#     print('Standardizing minimum and maximum temperatures...')
#     print(tair['tmax'].nbytes)
#     tair['tmax'] = standardize_values(tair['tmax'], tair['dates'], ndays)
#     tair['tmin'] = standardize_values(tair['tmin'], tair['dates'], ndays)
    
#     # Determinine the spatial mean of the min/max temperatures
#     print('Determining the spatial mean of the min/max temperatures')
#     tair['tmax_ts'] = np.nanmean(tair['tmax'].reshape(T, I*J, order = 'F'), axis = -1)
#     tair['tmin_ts'] = np.nanmean(tair['tmin'].reshape(T, I*J, order = 'F'), axis = -1)
    
#     with Dataset('temp_time_series.nc', 'w', format="NETCDF4") as nc:
#         nc.createDimension('time', size = tair['dates'].size)
        
#         nc.createVariable('dates', str, ('time', ))
#         nc.createVariable('tmean_ts', tair['tmean_ts'].dtype, ('time', ))
#         nc.createVariable('tmax_ts', tair['tmax_ts'].dtype, ('time', ))
#         nc.createVariable('tmin_ts', tair['tmin_ts'].dtype, ('time', ))
        
#         for n in range(len(tair['dates'])):
#             nc.variables['dates'][n] = str(tair['dates'][n])
            
#         nc.variables['tmean_ts'][:] = tair['tmean_ts'][:]
#         nc.variables['tmax_ts'][:] = tair['tmax_ts'][:]
#         nc.variables['tmin_ts'][:] = tair['tmin_ts'][:]
    
    # Refine the HW dates and add min/max temperature to the csv fiels
    print('Determining final HW start and end dates...')
    for n, percentile in enumerate(percentiles):
        filename = 'hwinitialdates_%s_%s_percentile.csv'%(region, percentile)   ######
        fn_new = 'hw_days_%s_percentile_%s.csv'%(percentile, region)       ######
        results = newHWevents(filename, './', fn_new, start_year, region, tair['dates'], tair, 'tmax_ts', 'tmax_ts') #######
        
        # Function does not set a minimum number of days for HW; set that minimum.
        print('Finding HWs with minimum number of consecutive days...')
        with open(fn_new, 'w') as file:
            file.write('Start_day,end_day,mean_T_max,mean_T_min' + '\n')
            for r in results:
                if r.size >= min_days[n]:
                    # file.write(tair['dates'][r[0]].strftime('%Y-%m-%d') + ',' + tair['dates'][r[-1]].strftime('%Y-%m-%d') + ',' + str(np.nanmean(tair['tmax_ts'][r])) + ',' + str(np.nanmean(tair['tmin_ts'][r])) + ',' + '\n')
                    file.write(tair['dates'][r[0]].strftime('%Y-%m-%d') + ',' + tair['dates'][r[-1]].strftime('%Y-%m-%d') + ',' + str(np.nanmean(tair['tmax_ts'][r])) + ',' + str(np.nanmean(tair['tmax_ts'][r])) + ',' + '\n')
            
        # Make some plots for each HW
        print('Making plots for each HW...')
        if len(lat) < 2:
            tair['lon'], tair['lat'] = np.meshgrid(tair['lon'], tair['lat'])

        for r in results:
            if r.size >= min_days[n]:
                # do_map(np.nanmean(tair['tmean'][r, :,:], axis=0), lat, lon,  r"Average $T_{mean}$ Anomaly %s for %s to %s"%('SGP',tair['dates'][r[0]].strftime('%b %d, %Y'), tair['dates'][r[-1]].strftime('%b %d, %Y')),
                #       var_name = r'$T_{mean}$ Anomalies', savename = "T_mean_%s_%s.png"%(region, tair['dates'][r[0]].strftime('%Y-%m-%d')))
                
                do_map(np.nanmean(tair['tmax'][r, :,:], axis=0)-273.15, lat, lon,  
                       r"Average $T_{max} (^\circ C)$ %s for %s to %s"%('SGP',tair['dates'][r[0]].strftime('%b %d, %Y'), tair['dates'][r[-1]].strftime('%b %d, %Y')),
                       var_name = r'$T_{max}$ Anomalies', savename = "T_max_%s_%s.png"%(region, tair['dates'][r[0]].strftime('%Y-%m-%d')))
                
                # do_map(np.nanmean(tair['tmin'][r, :,:], axis=0), lat, lon,  r"Average $T_{min}$ Anomaly %s for %s to %s"%('SGP',tair['dates'][r[0]].strftime('%b %d, %Y'), tair['dates'][r[-1]].strftime('%b %d, %Y')),
                #       var_name = r'$T_{min}$ Anomalies', savename = "T_min_%s_%s.png"%(region, tair['dates'][r[0]].strftime('%Y-%m-%d')))
                
        # Also obtain the average temperature maps for all HW events
        print('Making average T plots over all HWs...')
        result_new = []
        for r in results:
            if r.size >= min_days[n]:
                for value in r:
                    results_new.append(value)
                
        results_new = np.array(results_new)
        
        # do_map(np.nanmean(tair['tmean'][results_new, :,:], axis=0), lat, lon,  r"Average $T_{mean}$ Anomaly for all HWs", var_name = r'$T_{mean}$ Anomalies', savename = "T_mean_%s_all_HW.png"%region)
        do_map(np.nanmean(tair['tmax'][results_new, :,:], axis=0)-273.15, lat, lon,  r"Average $T_{max} (^\circ C)$ for all HWs", var_name = r'$T_{max}$ Anomalies', savename = "T_max_%s_all_HW.png"%region)
        # do_map(np.nanmean(tair['tmin'][results_new, :,:], axis=0), lat, lon,  r"Average $T_{min}$ Anomaly for all HWs", var_name = r'$T_{min}$ Anomalies', savename = "T_min_%s_all_HW.png"%region)