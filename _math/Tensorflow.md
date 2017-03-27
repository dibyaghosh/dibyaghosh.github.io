---
layout: post
title: "Intro to Measure Theory"
description: "A venture into the rabbit hole of probability theory"
date: January 05, 2017
featured: true
---
$$\def\RR{\mathbb{R}}$$
$$\def\NN{\mathbb{N}}$$
$$\def\cX{\mathcal{X}}$$
$$\def\cA{\mathcal{A}}$$

Measures
--------

We define a **measure** $\mu$ on as set ${\cX}$ to be a function which
assigns nonnegative values $\mu(A)$ to subsets $A$ of $\mathcal{X}$.
Here are some examples

1.  If $\mathcal{X}$ is countable, then we define $\mu(A)$ to be
    $$\mu(A) = |A|$$.

2.  If $\mathcal{X} = \mathbb{R}^n$, then we define $\mu(A)$ to be
    $${\int \dots \int}_A dx_1 \dots dx_n$$

    This is the *Lebesgue* measure on $\mathbb{R}^n$.

Now oftentimes, we cannot define such a function $\mu$ on all subsets of
$\mathcal{X}$. Instead, we limit ourselves to a subset of the power set
of $\mathcal{X}$.

**Definition:**

A $\mathbf{\sigma}$**-field** is a collection $\mathcal{A}$ of subsets
of a set $\mathcal{X}$ which satisfy the following

1.  $\cX \in \cA$

2.  If $S \in \cA$, then $S^c \in \cA$ (Closed under Complements)

3.  If $S_1 , S_2, \dots \in \cA$, then $\cup_{i=1}^\infty S_i \in \cA$
    (Closed under Countable Unions)

**Theorem:**

Some properties of a $\sigma$-field $\cA$ include

1.  $\emptyset \in \cA$

2.  $\cA$ is closed under countable intersections

**Definition:**

A **measure** $\mu$ on a $\sigma$-field $\cA $ of $\cX$ is a function
which satisfies

1.  $\mu: A \to [0,\infty]$

2.  (Additive over disjoint union)If $A_1, A_2, \dots \in \cA$ are
    disjoint ($i \neq j \implies A_i \cap A_j = \emptyset$), then

    $$\mu(\bigcup A_i) = \sum_{i=1}^\infty \mu(A_i)$$

**Theorem:**

Let $B_1,B_2 \dots$ be an increasing sequence (that is
$B_1 \subset B_2 \subset \dots $), with limit $B = \bigcup B_i$, then
$$\mu(B) = \lim_{n \to \infty} \mu(B_n)$$

*Proof:*

Consider the sequence of sets $(C_n)_{n \in \NN}$ defined as
$C_n = B_{n}\backslash B_{n-1}$, and $C_1 = B_1$. Notice that we have
$\cup_{i=1}^n C_i = B_n$, and further, $\cup_{i=1}^\infty C_i = B$.

$$\begin{aligned}

\mu(\bigcup C_i) &= \sum_{i=1}^\infty \mu(C_i)\\

\mu(B) &= \lim_{n \to \infty} \sum_{i=1}^n \mu(C_i)\\

\mu(B) &= \lim_{n \to \infty} \mu(B_i)\\\end{aligned}$$

We define a **measure space** to be a triple $(\cX,\cA,\mu)$. If there
are sets $A_1, A_2 \dots$ such that $\mu(A_i) < \infty ~~\forall i$ and
$\bigcap A_i = \cX$, then the measure is said to be *$\sigma$-finite*.
Further, if $\mu(\cX) < \infty)$, then the measure is said to be
*finite*.Finally, if $\mu(X) = 1$, then $(\cX,\cA,\mu)$ is called a
**probability space**

Integration
-----------

Here, we shape the definition of *integration* over a measure $\mu$.
First, we start with some motivating examples

1.  Counting Measure - $\int f d\mu = \sum_{x \in \cX} f(x)$

2.  Lebesgue Measure over $\RR^n$ -
    $\int f d\mu = \int \int \dots \int f(x_1, x_2, x_3 \dots x_n) dx_1 \dots dx_n$

**Definition:** A function $f:\cX \to \RR$ is **measurable** if for all
Borel sets $B$

$$f^{-1}(B) \in A$$

The basic properties of integrals are

1.  For all sets $A \in \cA$, $\int 1_A d\mu = \mu(A)$, where
    $1_A(x) = \begin{cases}1  & x \in A\\ 0 &otherwise\end{cases}$

2.  For all $ f,g$ nonnegative measurable functions, and $a,b > 0$, we
    have

    $$\int (af + bg) d\mu = a\int f d\mu + b\int g d\mu$$

3.  If $f_1 \leq f_2 \dots $ and $\lim f_n = f$, then
    $$\int f d\mu = \lim \int f_n d\mu$$

We say that a function $f$ is *simple* if it can be written as
$\sum_{i=1}^n a_i1_{A_i}$, where $a_i > 0$ and $A \in \cA$

**Theorem:** If $f$ is nonnegative and measurable, then there exists a
sequence of nonnegative nondecreasing *simple* functions
$f_1 \leq f_2 \dots$ such that $\lim f_n = f$

Events, Probabilities, and Random Variables
-------------------------------------------

Let’s take a closer look at probability spaces now, denoted by
$(\mathcal{E},\mathcal{B},P)$, where sets $B \in \mathcal{B}$ are called
*events*, points $e \in \mathcal{E}$ are called outcomes, and $P(B)$ is
called the *probability* of $B$.

**Definition:**

A measurable function $X : \mathcal{E} \to \RR$ is called a **random
variable**, and it induces a probability measure $P_X$, defined as

$$P_X(A) = P(\{e \in \mathcal{E} : X(e) \in A\})$$ The *cumulative
distribution function* of $X$ is defined by
$$F_X(x) = P(X \leq X) = P_X((-\infty,x]))$$

Null Sets
---------

A set $N$ is **null** with respect to a measure $\mu$ if $$\mu(N)=0$$

If a statement holds for $x \in \cX - N$ where $N$ is null, then we say
that the statement holds *almost everywhere* (a.e). Alternatively, if a
statement holds on $B$ such that $\mu(B) = 1$ (and thus $B^c$ is null),
then we say that the statement holds *with probability one*.

The value of an integral is unaffected by the values that it achieves on
null sets. As a result,

1.  If $f = 0$ (a.e. $\mu$), then $\int f d\mu = 0$

2.  If $f \geq 0$, and $\int f d\mu = 0$, then $f = 0$ (a.e. $\mu$)

3.  If $f = g$ (a.e. $\mu$), then $\int f d\mu = \int g d\mu$

Densities
---------

**Definition:**

Let $P$ and $\mu$ be measures on a sigma field $\cA $ of $\cX$. $P$ is
**absolutely continuous** w.r.t $\mu$, denoted as $P << \mu$,if
$P(A) = 0$ whenever $\mu(A) = 0$

**Theorem:**

**Radon-Nikodym** If a finite measure $P$ is absolutely continuous wrt a
$\sigma$-finite measure $\mu$, then there exists a nonnegative
measurable function $f$ such that $$P(A) = \int_A f d\mu$$.

The function $f$ is called the *Radon-Nikodym derivative* of $P$ wrt
$\mu$, or alternatively, the *density* of $P$ wrt $\mu$, and denoted as
$f = \frac{dP}{d\mu}$

Expectation
-----------

Let $X$ be a random variable on a probability space
$(\mathcal{E},\mathcal{B},P)$.

**Definition:**

The **expectation** of $X$ is given by $EX = \int X dP$. This formula
never being used in practice, we instead say that if $X \sim P_X$, that
$$EX = \int x~dP_X$$. Similarly, if $Y = f(X)$, then
$E(Y) = \int f(x) dP_X$. To compute integrals in practice, we say that
if $P_X$ has density $p$ wrt $\mu$, then $$\int f dP_x = \int fp d\mu$$

Using this definition of expectation, we further define the following
quantities:

1.  The **variance** of $X$ is given by $E(X-EX)^2$

2.  The **covariance** of $X$ and $Y$ is given by $E[(X-EX)(Y-EY)]$

Random Vectors
--------------

Let $X_1 \dots X_n$ be random variables on our generic probability
space. We define the function $X: \mathcal{C} \to \RR^n$ as
$$X(e) = \begin{bmatrix} X_1(e) \\ \vdots \\ X_n(e) \end{bmatrix}$$

We say that the random vector $X$ and it’s corresponding distribution
$P_X$ is *absolutely continuous* with density $p$ if $P_X$ is absolutely
continuous w.r.t the Lebesgue measure on $\RR^n$. We say that

$$P(X \in B) = {\int \dots \int}_B p(x) dx$$

The *expectation* of a random vector $X$ is the vector of it’s
expectations

$$EX = \begin{bmatrix} EX_1\\ \vdots \\ EX_n \end{bmatrix}$$

Covariance Matrices
-------------------

In what follows, $v$ is a constant vector, $A,B,C$ are constant
matrices, $X$ is a random vector, $W$ is a random matrix(A matrix whose
entries are random variables). These identities fall from linearity of
expectation

$$E[v + AX] = v + AE[X]$$

$$E[A + BWC] = A + BE[W]C$$

**Definition:**

The **covariance** of a random vector $X$ is the matrix such that the
entry $Cov(X)_{ij} = Cov(X_i,X_j)$ . We can denote this succinctly write
this as

$$Cov(X) = E[(X-\mu)(X-\mu)^T] = EXX^T - \mu\mu^T$$

We have that $cov(v + AX) = ACov(X)A^T$ by using the definition of
covariance and the formulas above

Product Measures and Independence
---------------------------------

We consider two distinct measure spaces $(\cX,\cA,\mu)$ and
$(\mathcal{Y},\mathcal{B},\nu)$

**Definition:**

The **product measure** is the unique measure $\mu \times \nu$ on
$(\cX \times \mathcal{Y}, \cA \lor \mathcal{B})$, such that
$$(\mu \times \nu)(A \times B) = \mu(A)\nu(B)$$

In the above definition, $A \lor B$ is the $\sigma$-field generated by
$\cA \times \mathcal{B}$ (the smallest $\sigma$-field containing
$A\times B$)

**Theorem:Fubini**

If $f \geq 0$, then

$$\int f d(\mu \times \nu) = \int \left[ \int f(x,y) d\nu(y)\right] d\mu(x) =  \int \left[ \int f(x,y) d\mu(x)\right] d\nu(y)$$

The following also holds if f isn’t constrained but
$\int |f| d(\mu \times \nu) < \infty$

<script type="text/bibliography">
  @article{gregor2015draw,
    title={Theano Tutorials},
    author={Radford, Alec},
    journal={github},
    year={2014},
    url={https://github.com/Newmu/Theano-Tutorials}
  },
@misc{tensorflow2015-whitepaper,
title={ <i>TensorFlow</i>: Large-Scale Machine Learning on Heterogeneous Systems},
url={http://tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart'in~Abadi  et al.},
  year={2015},
}
</script>