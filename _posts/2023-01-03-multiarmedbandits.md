---
title: "MultiArmedBandits with Stochastic, Adversarial, and Adversarially Corrupted Stochastic Rewards"
permalink: "/MultiArmedBandits/"
layout: post
mathjax: true
---

The multi-armed bandit (MAB) is a classic problem in probability theory and statistics that models exploitation-exploration trade-off, leveraging choices that have proven effective in the past versus choosing new options that might provide better-unexplored trade-offs. Imagine a row of slot machines, each with a different and unknown distribution of paying out a reward. The goal is to select the arm of a machine at each time instance such that it maximizes the total reward over time. This article will discuss some known algorithms in MAB in stochastically chosen, adversarially chosen, and stochastically chosen but strategically corrupted reward scenarios and run simulations for the algorithms discussed.

Notations used:
+ $$k$$ : number of arms / actions
+ $$[K]$$: the set all arms/actions
+ $$a \in K$$: a unique arm/action
+ $$T$$: total number of rounds
+ $$t \in [T]$$: a specific round
+ $$a_t$$ : arm chosen in round $$t$$
+ $$X_{a\in [K],t\in [T]}$$: reward for arm $$a$$ in round $$t$$ 
+ $$\Delta_{a,t}$$: is the difference between the reward of optimal arm and arm $$a$$ in round $$t$$.
+ $$n_t(a)$$: is the number of pulls of arm $$a$$ upto round $$t$$.

&nbsp;
#### Stochastic Bandits:

Stochastic bandits are type of problem in which the reward of different actions are unrelated to each other, and the rewards of the same action at different time steps are an i.i.d distribution. 
| **Protocol:** |
|------------------|
| ***Parameters***: $$K$$ arms, $$T$$ rounds, $$T > K$$ , for each arm $$a \in [K]$$, the reward for arm $$a$$ is drawn from distribution $$D_a$$. |
|For each round $$n \in [T]$$ the algorithm chooses an $$a_t\in[K]$$ and observes a reward $$X_{a_t,t}$$ |

The aim of the algorithm is to minimize the deficit suffered from not always choosing the arm, with the highest total expected reward.
Lets define this deficit as Pseudo-Regret, as: 
$$$
R = \max_{a\in [k]} \mathbb{E}\Big[\sum_{i\in[T]}X_{a,t} - \sum_{i\in[T]}X_{a_t,t} \Big]
$$$

For the purpose of this article, let's assume that all stochastic rewards are drawn from Normal Distribution with mean between 0 and 1.
Also, lets assume that the rewards are capped between -5 and 6.

<!--<sub><sup>Note: the probabilty that $$N(0,1) \in [-5,5] is > 10^6$$, so we can assume properties for both bounded and normal distribution as when required.</sub></sup>-->

##### Explore-then-commit:

Let's start with a simple algorithm: explore arms uniformly selecting each action $$N$$ times *(exploration phase)* the commiting to the best arm for the remaining $$T-NK$$ rounds *(exploitation phase)*.
This simple looking algorithm suffers from sub-linear i.e. o(T) regret.
Proof sketch: 
From [Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality), we can infer that if $$N$$ is chosen large enough, then the mean reward for each arm estimated by sampling in the exploration phase is almost equal to the true mean reward of the arm. Then the total regret for exploration rounds is bounded by $$NK$$ times $$\max_{a\in[K]}\Delta_a$$ , and the total regret in exploration rounds should be close tob negligiable.

Let $$\mu_a$$ be true mean of arm $$a$$ and $$\overline{\mu}_{a}$$ be the mean estimated by sampling. Then,
$$\implies\mathbb{P}\big[ | \mu_a - \overline{\mu}_a | < \delta \big] \geq 1 - e^{\frac{-2\delta^2}{N\cdot(6 - (-5 )^2)}} = 1 - e^{\frac{-2\delta^2}{N\cdot121}}$$
<!--Let $$\delta = \sqrt{\frac{2\cdot log(T)}{N}}$$-->
The the total regret suffered by the algorithm is the regret suffered in exploration phase + regret suffered in exploitation phase. The regret in any 1 round of exploration phase is bounded by the limits of rewards distribution *[6 - (-5)]* and with probability $$1 - e^{\frac{-2\delta^2}{N\cdot121}}$$ is no more than $$2\delta$$ and with probability $$e^{\frac{-2\delta^2}{N\cdot121}}$$ bounded by the limit of rewards distribution *[6 - (-5)]*
$$\implies R \leq N\cdot K\cdot 11 + (1 - e^{\frac{-2\delta^2}{N\cdot121}})\cdot2\delta + e^{\frac{-2\delta^2}{N\cdot121}}\cdot 11$$
 to minimize the above equation we can assume $$N$$ to be $$O\big((\frac{T^2\log T}{K^2})^\frac{1}{3}\big)$$ and  $$\delta$$ to be $$O\big(\sqrt{ \frac{2\log(T)}{N} }\big)$$
$$\implies R = O\big( T^{\frac{2}{3}}K^\frac{1}{3}(\log T)^\frac{1}{3} \big)$$


























&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

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
