---
title: "MultiArmedBandits with Stochastic, Adversarial, and Adversarially Corrupted Stochastic Rewards"
permalink: "/MultiArmedBandits/"
layout: page
mathjax: true
---

The multi-armed bandit (MAB) is a classic problem in probability theory and statistics that models decision-making scenarios where you need to choose between different options with uncertain rewards. This article will discuss some known algorithms in MAB in stochastically chosen, adversarially chosen and stochastically chosen but strategically corrupted rewards scenarios and run simulation for the algorithms discussed.




The Monty Hall Problem is one of the most nefarious problems in stochastic mathematical modelling. It is a simple looking problem that perplexes most people when confronted with its dilemma. This article will try to generalize the MHP on how the host of the show chooses his door to better understand the problem.


Let's consider, you are on a game show and the following happens:

1. You are presented with 3 doors(Let {$$A$$, $$B$$, $$C$$} ) in front of you and told that exactly one of them hides a prize and other two are empty your task is to select the door with the prize.
2. After you have made your selection, the host of the show selects a door that is not the same doors as yours.
3. The host's door is opened and it is found that, that door does not hide the prize.
4. You are offered to reconsider your option to the other unopened door or you can stick with your current door, before your door is opened and it is revelaed whether your task was succesful.


What should you do? which door has higher chances of containing  the prize.


The above problem is a slightly perturbed version of the MHP, in the monty hall problem, you are given with a crucial piece of inforamtion about step 2 i.e. how the host chooses his door.


Consider a not-Monty Hall problem , where the host is also unaware of the location of the prize and randomly chooses one of the other doors you haven't chosen. When you observe that this door does not contain the prize, how does your random variable update with this new information. (Lets say you had chosen the door A)


This new probability distribution can be easily calculated by the the concept of Conditional Random Variable.


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


In this case, the probability of your previously selected door containing the gift is $$1/3$$, and hence, you chances of winning are higher if you choose to switch.





Lets verify this argument via a simulation, in case of monty hall and not-monty hall problem.


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

Run the above simulation a large number of times to convince of correct probabilities.

{% highlight python %}
ans = [0,0]
for i in range(10000):
	t = (not_)monty_hall_sim()
	ans[0] += t[0]
	ans[1] += t[1]
print("Winning Prob if not switching: " , ans[0]/ans[1])
{%endhighlight%}
