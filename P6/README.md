# Summary

In this project I have studied the intermediate dataset [Flights](https://www.google.com/url?q=http://stat-computing.org/dataexpo/2009/the-data.html&sa=D&ust=1454271917244000&usg=AFQjCNEo7P1zBM-dtkX-MwsZiev7-J1MRw). In particular, I have investigated the performance of US carriers in the year of [2008](http://stat-computing.org/dataexpo/2009/2008.csv.bz).

The data comes originally from [RITA](http://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp). Detailed information about the dataset can be found on the RITA website. The website hosts flight performance between 1987 and 2008. The data can be downloaded in the website.

My visualization can be seen at http://rawgit.com/ruimaranhao/DataAnalystNanodegree/master/P6/index.html.

### Variable descriptions

| Name	| Description |
| ----- |-------------|
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

After exploring the data, I decided to visualize the average delays per airport
(v1) as a bar plot. Not being quite happy with the visualization, I have decided
to plot two things: route average delays (chord diagram) and average delays per
carrier (barplot). This visualization are in v2.

After internal feedback, I decided to upgrade the chord diagram to include the
routes between the top 10 busiest airports. This drastically simplified the
visualization.

Finally, after all feedback, I have included a legend to people understand the
different elements in the chord diagram and I have changed the bar plot to a
stack chart diagram.

# Feedback

I have shared my visualization on Facebook, Google plus and the Nanodegree
class discussion. I have got feedback such as:


>Uirá (udacity mentor): Nice start. I wonder what are all these colors in your first chart. Perhaps you could use a legend here or tooltips to clarify the information encoded. In your second chart, if it is your desire to show the most common reasons for delays, maybe would be more appropriate using another kind of visualization ( a stacked chart, perhaps).
Another suggestion: Remember that one of the key goals of this project is to "communicate a story ." So, for example, you could include some text or animation highlighting some interesting findings that the users can observe in your visualization.

>Matthew Klenk (facebook): They are pretty but I'm not sure what I'm suppose to take away from the graph on the left.
>Matthew Klenk (facebook): Cool, I'm still not sure the question. I feel the like the question you want an answer to is, what is the most likely delayed routes. And part of this would then be what is the distribution of delayed likelihoods. Are they all about the same or are there some that are just awful? I'd prefer a histogram for this, but perhaps that is what the colors are for. I guess in this plot you get two controls, the width of the connections and the color. So I guess the width is how many flights there are, and the color is how delayed they are?

>Peter Bunus (facebook): Cool! I like both graphs. It is clear to me what you are trying to show. Are you using the data from the US Department of Transportation?

The common concern was how to read the chord diagram. I have addressed that by
adding a short explanation to tell the story.

I have had multiple discussions regarding the diagrams with other people that
helped me clarify the visualizations/sketches.

After the feedback I thought about implementing something similar to
http://datamaps.github.io/. I've started that implementation in v3, but have
not concluded it yet.

# Resources

You will need to run local server to render the visualizations. You can start a local server using Python. Navigate to the directory that contains all of the files and then type `python -m SimpleHTTPServer` in the command line. If you type `http://localhost:8000` into the address bar of your browser, then you should see the files that you can display in the web page.

- http://d3js.org/

- http://dimplejs.org/

- http://stackoverflow.com/

- Many other websites found through Google.
