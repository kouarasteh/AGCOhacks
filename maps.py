
api_key = 'API_KEY'

ll_list = [(40.207931, -88.315668),(40.242104, -88.242056),(40.109433, -88.119437),(40.098775, -88.100009),(40.069705, -88.114210),(40.047706, -88.114304)]

https://maps.googleapis.com/maps/api/staticmap?center=40.714728,-73.998672&zoom=12&size=400x400&key=YOUR_API_KEY




def lltoPix(minlat,minlong,maxlat,maxlong,lat,long,fw,fh,fpwidth,fpheight):
    rellat = lat-minlat
    rellong = long-minlong
    pplat = fh/abs(maxlat-minlat)
    pplong = fw/abs(maxlong-minlong)
    x = int(pplong*rellong)
    y = int(pplat*rellat)
    print(x,y)
    return(x,y)
