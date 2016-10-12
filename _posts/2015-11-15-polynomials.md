---
layout: post
title: Interpolating Polynomials 
description: Using adaptive queries to interpolate any integer polynomial in 2 or less queriescd 
---

I've been reading up on polynomials and their properties lately, and a certain question caught my eye . 

>Adam and Betty are playing a game related to guessing polynomials. Adam has thought up a polynomial (no limit on degrees) with all positive integer coefficients, which we call P(x). Betty can ask for the values of the polynomial at any integer x, and using only this information, she must discern what Adam's polynomial is. What is the maximum number of queries that Betty will have to make? 

Your first inclination might be that the number of queries depends on the degree of the polynomial, but I'll make the claim that Betty will only require 2 guesses to guess Adam's polynomial. The key step that allows us to discern an nth degree polynomial in constant queries is a process called adaptive querying. By using the result of the first query, we can determine a value x, which will allow us to extract the values of the coefficients, no matter the degree of  the polynomial.
###### The Game of Adaptive Queries

Let's find a strategy so that it will only take 2 queries to solve the problem. We can represent Adam's polynomial as, 

$$p(x) = a\_0 + a\_1x + a\_2x^2 + \dots + a\_nx^n $$

where $a\_0, a\_1, a\_2 \dots a\_n$  represent the coefficients of the numbers. Notice the similarity between the formula for this polynomial, and the representation of a number in base_x. For instance, a number given as $a\_1a\_0$ in base 10 can be rewritten as $a\_0 + 10*a\_1$. Similarly, a number written as $a\_2a\_1a\_0$ in base 5 can be rewritten as $a\_0 + a\_1(5)^2 + a\_2(5)^3$.Since the coefficients of Adam's coefficients are all positive, his polynomial is analogous as rewriting a number from base $x$. 

$$ a\_0 + a\_1x + a\_2x^2 + \dots + a\_{n-1}x^{n-1} + a\_nx^n = a\_na\_{n-1} \dots a\_2a\_1a\_0 ~~(base x)$$

We must be careful though, since this property will only hold when x is larger than all of the coefficients (otherwise we might have carrying over, which distorts the data). How do we pick $x$ then? We now must find a way to find the value of the highest possible coefficient. Notice that $P(1)$ is simply $a\_0 + a\_1 + a\_2 + \dots + a\_n$, which is always greater than or equal to the largest coefficient (since all coefficients are positive). Taking x to be $P(1)+1$ guarantees that we now have a base that will properly encode the values of the coefficients. The final step remains to convert this number to base $x+1$.

###### Time to Code

To summarize the previous paragraph, our strategy is to first query for $P(1)$, then $P(~P(1) +1~) $

I'll be coding this example in Python: the choice of language is arbitrary, so feel free to use whatever language suits your fancy. Some structural code: the function _make\_polynomial_ creates a polynomial f(x) whose coefficients are that of the list that you passed in, and the function _tobase_ converts a number into certain base (outputs a list of digits). 
```python
def make_polynomial(coefficients):  
	"""Given a list of coefficients: [a0, a1, a2, ... an], returns a polynomial
		function which computes the value of the polynomial at any given x"""
	def f(x):
		value = 0
		for n,a in enumerate(coefficients):
			value += a*(x**n) 
		return value
	return f

def tobase(x,num):
	""" Converts a num to base x"""
	coefficients = []
	while num > 0:
		coeff = num % x
		coefficients.append(coeff)
		num = num // x
	return coefficients
```   
Our strategy is now quite simple:

```python
def guess_coefficients(p):
	""" Given a polynomial function p, finds its coefficients """
	x = p(1)+1
	num = p(x)
	return tobase(num,x)
``` 

#### Taking it Further

Now let's look at a variant of this problem, except a tad harder:

>Adam has now thought of a polynomial (of nth degree), whose coefficients are real numbers. Betty can ask for the values of the polynomial at any integer x, and using only this information, she must discern what Adam's polynomial is. What is the maximum number of queries that Betty will have to make? 

Now the coefficients are no longer positive integers, so our base-changing strategy will no longer work. So let's start afresh: simple intuition leads us to believe that it should take n+1 points to determine this new polynomial. After all, a 0 degree polynomial ($y=a\_0$) is uniquely identified by one point, and a 1 degree polynomial (a line $y = a\_0 + a\_1x$) by 2 points. Let's try to put mathematics behind this intuition:

**Theorem: A polynomial of degree $n$ is uniquely identified by $n+1$ points**

__Proof:__

Let $P_n$ represent the set of all polynomials with real coefficients up to degree n

Let T be a linear transformation given by 
 
$$ T(P) = \begin{bmatrix} P(0) \cr P(1) \cr P(2) \cr \vdots \cr P(N) \end{bmatrix} $$

Our goal is to prove that the linear transformation $T(p)$ is an injection from the set of polynomials to $R^{n+1}$. In other words, any polynomial can be represented by a unique vector in $R^{n+1}$

First let us observe the kernel of this transformation. Remember that the kernel of a tranformation is a subset of the domain which maps to the zero element of the codomain. The only polynomial in $P\_n$ that maps to the zero vector is $0$, since $T(0) =  \begin{bmatrix} P(0) \cr P(1) \cr P(2) \cr \vdots \cr P(N) \end{bmatrix} $ leads to the zero vector, $\begin{bmatrix} 0 \cr 0 \cr 0 \cr \vdots \cr 0 \end{bmatrix} $. Since the polynomial $ p(x) = 0$ is the only element that maps to zero, we can consider the kernel (null space) to be trivial. By definition, since the kernel is trivial, the transformation is injective, and our theorem is proven. 

