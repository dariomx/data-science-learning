---
output:
  pdf_document: default
  html_document: default
---
Show that 

\[
u_i = \frac{1}{\sigma_i} A v_i
\]

Let us begin with the SVD factorization, that we can always apply:

\[
A = U D V^T
\]

Let us now multiply both sides on the right by $V$ (whose inverse is $V^T$ due its orthogonality):

\[
A V  = U D 
\]

Above matrix equality implies in particular, the equality column-wise (we are reusing same notation as previous exercise, for projecting a particular column; which is equivalent to the lower-case notation using in notes, that is $[M]_i = m_i$):

\[
[A V]_i  = [U D]_i
\]

We reuse same result from previous exercise, where a matrix product can be seen as linear combinations over the columns of left operand. In particular, each column of the product is the linear combination of the left-matrix columns, using as coefficients the values of a fixed right-matrix column. Applying this to the left hand of the equality yields:

\[
[A V]_i  = \sum_{k=1}^n [A]_k V_{ki} = A v_i
\]

Applying the same reasoning to the right side of equation, leads to the following simplification; whose last step is possible due the diagonal nature of matrix $D$:

\[
[U D]_i = \sum_{k=1}^n [U]_k D_{ki} = u_i \sigma_i
\]

Putting together (transitivity of equality), leads to:

\[
A v_i = u_i \sigma_i
\]

which we can easily re-arrange as:

\[
u_i = \frac{1}{\sigma_i} A v_i
\]

proving the result.

