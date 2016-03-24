# Summary

In this project I have studied the intermediate dataset [Flights](https://www.google.com/url?q=http://stat-computing.org/dataexpo/2009/the-data.html&sa=D&ust=1454271917244000&usg=AFQjCNEo7P1zBM-dtkX-MwsZiev7-J1MRw). In particular, I have investigated the the reasons for flight cancellations per month in  [2008](http://stat-computing.org/dataexpo/2009/2008.csv.bz).

The data comes originally from [RITA](http://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp). Detailed information about the dataset can be found on the RITA website. The website hosts flight performance between 1987 and 2008. The data can be downloaded in the website.

The visualization can be seen at http://bl.ocks.org/ruimaranhao/raw/3155b6bf1f832835cb31/ ([gist](https://gist.github.com/ruimaranhao/3155b6bf1f832835cb31), [http://bl.ocks.org/](http://bl.ocks.org/ruimaranhao/3155b6bf1f832835cb31)). The visualization shows that flight cancellations are more prevalent in the winter months (January, February, December). Severe flight cancellations are usually because of extreme weather conditions and/or the carriers themselves. National Airspace System Delays are also prevalent, but in many cases these delays are also related to (non-extreme) weather conditions. (it also include: airport operations, heavy traffic volume, air traffic control).

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

For the first submission, after all feedback, I have included a legend to people
understand the different elements in the chord diagram and I have changed the bar
plot to a stack chart diagram.

As the the evaluation feedback, pointed out at the there was not a clear, specific
finding in either visualization, I decided to study the dataset further to be able
to produce an explanatory chart with a clear, interesting finding. In particular,
I studied reasons for flight cancellations in 2008.

I decided to plot the information regarding flight cancellations using a Storyboard
Control plot, using the [dimplejs](http://dimplejs.org/advanced_examples_viewer.html?id=advanced_storyboard_control),
where each bubble means particular air carrier.

# Feedback

I have shared my visualization on Facebook, Google plus and the Nanodegree
class discussion. I have got feedback such as:

>Uirá (udacity mentor): Nice start. I wonder what are all these colors in your first chart. Perhaps you could use a legend here or tooltips to clarify the information encoded. In your second chart, if it is your desire to show the most common reasons for delays, maybe would be more appropriate using another kind of visualization ( a stacked chart, perhaps).
Another suggestion: Remember that one of the key goals of this project is to "communicate a story ." So, for example, you could include some text or animation highlighting some interesting findings that the users can observe in your visualization.

>Matthew Klenk (facebook): They are pretty but I'm not sure what I'm suppose to take away from the graph on the left.
>Matthew Klenk (facebook): Cool, I'm still not sure the question. I feel the like the question you want an answer to is, what is the most likely delayed routes. And part of this would then be what is the distribution of delayed likelihoods. Are they all about the same or are there some that are just awful? I'd prefer a histogram for this, but perhaps that is what the colors are for. I guess in this plot you get two controls, the width of the connections and the color. So I guess the width is how many flights there are, and the color is how delayed they are?

>Peter Bunus (facebook): Cool! I like both graphs. It is clear to me what you are trying to show. Are you using the data from the US Department of Transportation?

>John Enyeart (Google plus):  That's really cool. I especially like the visualization on the left.
Some things I noticed that might be improved:
The visualizations don't fit in my browser window unless I maximize it, so an auto-resize to make them smaller for smaller browser windows might be nice.
The graph on the right feels a little cluttered. Also, horizontal grid lines might make it easier to read the histogram values more to the right.﻿
The common concern was how to read the chord diagram. I have addressed that by
adding a short explanation to tell the story.

>Shi Shu (Google plus): Both figures are very cool! I am not sure what the animation of the bar chart (the right one) tells me. ﻿

I have had multiple discussions regarding the diagrams with other people that
helped me clarify the visualizations/sketches. Despite the encouraging feedback
from peer students, colleagues, and friends, the review from the Udacity's teacher
indicated that the plot need a significant change:

>Both of these charts are aesthetically pleasing and technically complex. Great job coding them.
I can't find a clear, specific finding in either visualization. The charts come across as exploratory rather than explanatory. An explanatory chart has a clear, interesting finding that would only be found through doing some data analysis. An exploratory visualization generally plots the data as is and leaves the reader to figure out a story and analyze the data for him or herself.

Given this feedback, I decided analyze the data and find something interesting to say about the data set. I decided to tell the story about the most common reasons for flight cancellations. The current x-axis is `Total Flights`, and this was the outcome of discussions with colleagues (the other option I had in mind was `Distance`, but it did not convey a clear message).

After another round of feedback, the I have updated the plot as follows:

1. I converted the legend into percentage values so we will be able to see what month has more delays percentage
2. I (partially) avoided X scaling, by just making chart two times bigger

I've got feedback from another colleague and updated the plot accordingly: change to a different color when the month is selected (and animation is paused).

# Summary of Findings

>Severe flight cancellations are usually because of extreme weather conditions and/or the carriers themselves. National Airspace System Delays are also prevalent, but in many cases these delays are also related to (non-extreme) weather conditions. (it also incude: airport operations, heavy traffic volume, air traffic control). Security Delays seldom occur.

I further investigated the fact that American Airlines had a 90% Carrier Delay in April 2008 (I decided to look into this because many flights where cancelled and American Airlines is one of biggest airliners in the world). I found that American Airlines has canceled thousands of flights for safety checks on its passenger planes. The FAA says the jetliners hadn't been properly inspected, and several other U.S. carriers have had to cancel flights as well. To get through the logistical chaos, the airlines are shuffling passengers, empty planes, mechanics, inspectors — and a lot of paperwork. ([source](http://www.cnn.com/2008/TRAVEL/04/10/american.cancellations/))

# Resources

You will need to run local server to render the visualizations. You can start a local server using Python. Navigate to the directory that contains all of the files and then type `python -m SimpleHTTPServer` in the command line. If you type `http://localhost:8000` into the address bar of your browser, then you should see the files that you can display in the web page.

List of resources I referred to:

- http://d3js.org/

- http://dimplejs.org/

- http://stackoverflow.com/

- Many other websites found through Google.
