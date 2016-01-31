# Summary

In this project I have studied the intermediate dataset [Flights](https://www.google.com/url?q=http://stat-computing.org/dataexpo/2009/the-data.html&sa=D&ust=1454271917244000&usg=AFQjCNEo7P1zBM-dtkX-MwsZiev7-J1MRw). In particular, I have investigated the performance of US carriers in the year of [2008](http://stat-computing.org/dataexpo/2009/2008.csv.bz).

The data comes originally from [RITA](http://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp). Detailed information about the dataset can be found on the RITA website. The website hosts flight performance between 1987 and 2008. The data can be downloaded in the website.

## Variable descriptions

| Name	| Description |
| ----- |:-----------:|
|	Year	| 1987-2008   |
|	Month	| 1-12        |
|	DayofMonth |	1-31  |
|	DayOfWeek	| 1 (Monday) - 7 (Sunday) |
|	DepTime	| actual departure time (local, hhmm) |
|	CRSDepTime |	scheduled departure time (local, hhmm) |
|	ArrTime | actual arrival time (local, hhmm) |
|	CRSArrTime |	scheduled arrival time (local, hhmm) |
|	UniqueCarrier |	unique carrier code |
|	FlightNum	| flight number |
|	TailNum	| plane tail number |
|	ActualElapsedTime |	in minutes |
|	CRSElapsedTime |	in minutes |
|	AirTime	| in minutes |
|	ArrDelay |	arrival delay, in minutes |
|	DepDelay |	departure delay, in minutes |
|	Origin |	origin IATA airport code |
|	Dest |	destination IATA airport code |
|	Distance |	in miles |
|	TaxiIn |	taxi in time, in minutes |
|	TaxiOut	| taxi out time in minutes |
|	Cancelled |	was the flight cancelled? |
|	CancellationCode |	reason for cancellation (A = carrier, B = weather, C = NAS, D = security) |
|	Diverted |	1 = yes, 0 = no |
|	CarrierDelay |	in minutes |
|	WeatherDelay |	in minutes |
|	NASDelay |	in minutes |
|	SecurityDelay |	in minutes |
|	LateAircraftDelay |	in minutes |

# Design

# Feedback

# Resources

You will need to run local server to render the visualizations. You can start a local server using Python. Navigate to the directory that contains all of the files and then type `python -m SimpleHTTPServer` in the command line. If you type `http://localhost:8000` into the address bar of your browser, then you should see the files that you can display in the web page.
