---
title: "Monty Hall, A Misunderstood Problem"
permalink: "/The_Monty_Hall_Problem/"
layout: page
mathjax: true
---


The Monty Hall Problem is a well-known puzzle in probability that may seem straightforward at first but is surprisingly tricky and confusing. This article will delve into the problem by examining how the host's choice of door affects the game, to better understand the problem.



Let's consider, you are on a game show and the following happens:

1. You are presented with 3 doors(Let {$$A$$, $$B$$, $$C$$} ) in front of you and told that exactly one of them hides a prize and the other two are empty your task is to select a door.
2. After you have made your selection, the host of the show selects a different door.
3. The host's door is opened and it is revealed that that door does not hide the prize.
4. You are offered to reconsider your option, you can select the other unopened door or you can stick with your current door.


What should you do? which door has higher chances of containing  the prize.


The above problem is a slightly perturbed version of the MHP, in the monty hall problem, you are given with a crucial piece of inforamtion about step 2 i.e. how the host chooses his door. In MHP you are told that the host is aware of the price door and deliberately chooses the other door without the prize.


Now, consider a not-Monty Hall problem, where the host is also unaware of the location of the prize and randomly chooses one of the other doors. The remaining steps remain the same.

Let's run a simulation, for monty hall and not-monty hall problem to see the probability of winning with changing the door choice.

For the not-monty hall problem:
{% highlight python %}

def not_monty_hall_sim():
	doors = ['a','b','c']
	#Let the prize be randomly chosen
	prize = random.choice( doors )
	#Your randomly chosen initial guess
	contestant_choice = random.choice(doors)
	#Host randomly choosing one of the remaining doors.
	host_choice = random.choice(list(filter(lambda x:x!=contestant_choice, doors)))
	#This is to satify the conditional information given to us.
	# since we know that when the host choice was revealed it was not the prize doors
	if (  host_choice == prize ):
		# Abort the simulation
		return (0,0)
	#Returing the output, the first index is if the inital guess was correct, and the second index is if this is a valid iteration. 
	if 	contestant_choice != prize:
		return (1 , 1)
	else:
		return (0 , 1)	
{% endhighlight %}


For the standard monty hall problem

{% highlight python %}
def monty_hall_sim():
	doors = ['a','b','c']
	#Let the prize be randomly chosen
	prize = random.choice( doors )
	#Your randomly chosen initial guess
	contestant_choice = random.choice(doors)
	#Host randomly choosing one of the remaining doors, which is not the prize door
 	#Since, host is aware of the prize door and is deliberately choosing one that is not the prize door.
	host_choice = random.choice(list(filter(lambda x:x!=contestant_choice and x!= prize, doors)))
	#This is to satify the conditional statement given to us.
 	# since we know that when the host choice was revealed it was not the prize doors
	if (  host_choice == prize ):
		# Abort the simulation
		return [0,0]
	#Returing the output, the first index is if the inital guess was correct, and the second index is if this is a valid iteration. 
	if 	contestant_choice != prize:
		return [1 , 1]
	else:
		return [0 , 1]	
{%endhighlight%}

After running the above simulation for 10,000 times, the probability of winning with changing the doors is:
For Monty Hall Problem : 0.6653 $$\approx \frac{2}{3}$$
For Not-Monty Hall Problem: 0.49126  $$\approx \frac{1}{2}$$

Let's discuss the above result:

When the contestant observe's that the host's door does not contain the prize, how does the random variable(of prize door) update with this new information.
Lets say the contestant had chosen the door A.


This new probability distribution can be calculated by the the concept of Conditional Random Variable.

For the not-monty hall problem, 

$$=P( A \text{ is prize | One of randomly chosen B/C door is empty} )$$
$$= \frac{P( A \text{ is prize} \cap\text{ One of randomly chosen B/C door is empty}  )}{P(\text{One of randomly chosen B/C door is empty})}$$ 

$$= \frac{P( A\text{ is prize})}{P(\text{One of randomly chosen B/C door is empty})}$$

$$= \frac{1/3}{2/3} = 1/2$$

And thus, in this case it does not matter if you choose to switch or not since either way your winning odds are at half.


In the actual monty hall problem, the host is aware of the prize and delibrately chooses a door which does not contain the prize. In this case, 


$$= P( A \text{ is prize | there exists a door not A, that does not have the prize} )$$

$$= \frac{P( A\text{ is prize }\cap\text{ there exists a door not A, that does not have the prize } )}{P(\text{there exists a door not }A\text{, that does not have the prize})}$$

$$= \frac{P( A\text{ is prize})}{P(\text{there exists a door not A, that does not have the prize})}$$ 

$$= \frac{1/3}{1} = 1/3$$

In this case, the probability of your previously selected door containing the reward is $$1/3$$, and hence, you chances of winning are higher if you choose to switch.
