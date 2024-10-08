---
title: "Monty Hall, A Misunderstood Problem"
permalink: "/The_Monty_Hall_Problem/"
layout: page
mathjax: true
---


The Monty Hall Problem is a classic puzzle in probability theory that may seem straightforward at first but is surprisingly tricky and confusing. This article will delve into the problem by examining how the host's choice of door affects the game, to better understand the problem.



### The Problem

You are on a game show with 3 doors labeled $$\{A, B, C\}$$. Behind one door is a prize, and behind the other two doors are empty. Here’s how the game works:

1. You choose one of the doors.
2. The host selects a different door.
3. The host's door is opened and it is revealed that that door does not hide the prize.
4. You are offered to reconsider your option, you can select the other unopened door or you can stick with your current door.

What should you do to maximize your chances of winning?

The above problem is a slightly perturbed version of the MHP, in the Monty Hall problem, you are given a crucial piece of information about step 2 i.e. how the host chooses his door. In MHP you are told that the host is aware of the price door and deliberately chooses the other door without the prize.

Now, consider a Not-Monty Hall problem, where the host is also unaware of the location of the prize and randomly chooses one of the other doors. The remaining steps remain the same.

Let's run a simulation, for the Monty Hall and the Not-Monty Hall problem to see the probability of winning with changing the door choice.

For the not-monty hall problem:
{% highlight python %}
import random
import numpy as np
{% endhighlight %}
{% highlight python %}
def not_monty_hall_sim():
	doors = ['a','b','c']
	#Let the prize be randomly chosen
	prize = random.choice( doors )
	#Your randomly chosen initial guess
	contestant_choice = random.choice(doors)
	#Host randomly choosing one of the remaining doors.
	host_choice = random.choice(list(filter(lambda x:x!=contestant_choice, doors)))
	#This is to satisfy the conditional information given to us.
	# since we know that when the host choice was revealed it was not the prize doors
	if (  host_choice == prize ):
		# Abort the simulation
		return (0,0)
	#Returing the output, the first index is if the initial guess was correct, and the second index is if this is a valid iteration. 
	if 	contestant_choice != prize:
		return (1 , 1)
	else:
		return (0 , 1)	
{% endhighlight %}


For the standard Monty Hall problem

{% highlight python %}
def monty_hall_sim():
	doors = ['a','b','c']
	#Let the prize be randomly chosen
	prize = random.choice( doors )
	#Your randomly chosen initial guess
	contestant_choice = random.choice(doors)
	#Host randomly choosing one of the remaining doors, which is not the prize door
 	#Since, the host is aware of the prize door and is deliberately choosing one that is not the prize door.
	host_choice = random.choice(list(filter(lambda x:x!=contestant_choice and x!= prize, doors)))
	#This is to satisfy the conditional statement given to us.
 	# since we know that when the host choice was revealed it was not the prize doors
	if (host_choice == prize):
		# Abort the simulation
		return [0,0]
	#Returing the output, the first index is if the initial guess was correct, and the second index is if this is a valid iteration. 
	if contestant_choice != prize:
		return [1 , 1]
	else:
		return [0 , 1]	
{%endhighlight%}

After running the above simulation 10,000 times, the probability of winning with changing the doors is:
{% highlight python %}
ans = np.array([0,0])
for t in range(10000):
	ans += not_monty_hall_sim()
print(ans[0]/ans[1])
{%endhighlight%}

For Monty Hall Problem : 0.6653 $$\approx \frac{2}{3}$$

For Not-Monty Hall Problem: 0.49126  $$\approx \frac{1}{2}$$

Let's discuss the above result:

When the contestant observes that the host's door does not contain the prize, how does the random variable(of prize door) update with this new information?
Let's say the contestant had chosen door A.


This new probability distribution can be calculated by the concept of Conditional Random Variable.

For the not-monty hall problem, 

$$=P( A \text{ is prize | One of randomly chosen B/C door is empty} )$$

$$= \frac{P( A \text{ is prize} \cap\text{ One of randomly chosen B/C door is empty}  )}{P(\text{One of randomly chosen B/C door is empty})}$$ 

$$= \frac{P( A\text{ is prize})}{P(\text{One of randomly chosen B/C door is empty})}$$

$$= \frac{1/3}{2/3} = 1/2$$

Thus, in this case, it does not matter if you choose to switch or not since either way your winning odds are at half.


In the actual Monty Hall problem, the host is aware of the prize and deliberately chooses a door that does not contain the prize. In this case, 


$$= P( A \text{ is prize | there exists a door not A, that does not have the prize} )$$

$$= \frac{P( A\text{ is prize }\cap\text{ there exists a door not A, that does not have the prize } )}{P(\text{there exists a door not }A\text{, that does not have the prize})}$$

$$= \frac{P( A\text{ is prize})}{P(\text{there exists a door not A, that does not have the prize})}$$ 

$$= \frac{1/3}{1} = 1/3$$

In this case, the probability of your previously selected door containing the reward is $$1/3$$, and hence, the contestant's chances of winning are higher if they choose to switch.
