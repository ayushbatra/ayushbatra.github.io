---
title: "Insurance? A Dummy example"
permalink: "/Insurance/"
layout: page
mathjax: true
---

Lets try to understand what insurance means from the hypothetical example of Alex.<br />
Alex one day wakes to find a magic coin with the following instructions:

* You are only allowed to flip the coin once and it will vanish afterwards.
* The coin has an equal probability of landing either 'Heads' or 'Tails'.
* If the coin lands on 'Heads' you will be given 10 million. 
* If the coin lands on 'Tails' you will be given nothing.







  

![Rules](/assets/story.png "Rules")


Now Alex is very happy at the prosprect of earning 10 million but is not ready to accept that this deal is not guaranteed and there is a 0.5 probability that he may not get anything. So instead he goes to Alpha Insurance company and offers them the following deal:
<br />
He will flip the magic coin and it lands on head he will share 6 million from his 10 million winnings but if the coin lands tails, the insurance company will have to pay him 4 million.
<br />
The company accepts the deal thinking this is a net positive for their profits and Alex walks out with 4 million irrespective of what sides the coin lands.
<br /><br />

This hypothetical example illustrates how insurance works in a simplified and ideal manner.

## Who gets what?

Lets try to understand how Alpha Insurance or Alex would benefit from a deal like this and why would they rationally choose to participate.


### Alpha Insurance
Alpha insurance would participate in this deal because this is an expected positive for them a simple calculation would suggest

$$E[money] = \frac{1}{2}\cdot 6 Million + \frac{1}{2}\cdot-1 * 4 Million = 1 Million$$


that on average they are profiting by 1 million, and since they are in market to maximise their profits they took this deal.


### Alex
Since this deal is an expected positive money for Alpha Insurance, and the net amount of money is constant, this deal is an expected negative money for Alex.

$$E[money] = \frac{1}{2}\cdot -1 * 6 Million + \frac{1}{2} \cdot 4 Million = -1 Million$$


Why would Alex participate in this deal if he knows that he is losing money. It is because he is not trying to maximise exopected amount of money he is trying to maximze another attribute - lets call it utility - that expresses the value this money brings in his life. 

<img src="/assets/Law_of_Dim_Mar_U.png" alt="Law of Diminishing Marginal Utility" width="700"/>

This phenomenon is understood by the "Law of Diminishing Marginal Utility" which implies that the value of something gets lower if you have a large amount of this thing. In this case money, Alex would prefer having more money as depicted by the increasing function of utility with money but this value increases less and less when he has larger and larger amount of money.<br /> In other words the value that he expects to gain in his life by getting 4 million is more than the value he expects to lose in his life by losing 6 million. Thus he has a positive expected value this deal and would choose to participate in this stochastic transaction with Alpha Insurance.



<br />
<br />

In slightly different words, an insurance service could be considered a courier service to send money from your more prosperous future to a your less prosperous future.
