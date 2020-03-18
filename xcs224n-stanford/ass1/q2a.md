---
output:
  pdf_document: default
  html_document: default
---
Show that 

\[
A = \sum_{i=1}^r \sigma_i u_i v_i^{T}
\]

Our strategy will leverage the SVD factorization, that is, above statement is equivalent to prove this equality:

\[
U D V^T = \sum_{i=1}^r \sigma_i u_i v_i^{T}
\]

And with above equality as the goal, we will use the strategy of showing the equality element-wise. Let us begin with the left side, in concrete with the product $D V^T$.

We will use a result from Linear Algebra, which says that a matrix product can be expressed as linear combinations over the columns of the left operand. Honoring the convention that the i-th right subscript of a matrix variable means its i-th column; this property can be written as follows (note also that we are using square brackets prior projecting by column, for the sake of clarity; such that we do not confuse that operation with regular parentheses grouping):

\[
[A B]_j = \sum_{k} [A]_k B_{kj}
\]

Applying above result to our product $D V^T$ yields:

\[
[D V^T]_j = \sum_{k}^r [D]_k V^T_{kj} = \sum_{k}^r [D]_k V_{jk}
\]

Now, given that $D$ is a diagonal matrix, the only element that really counts from each column $[D]_k$ is actually $D_{jk} = \sigma_k$, which means that:

\[
\sum_{k}^r [D]_k V_{jk} = 
(\sigma_1 V_{j1}, \sigma_2 V_{j2}, \dots, \sigma_r V_{jr})^T
\]

Hence, we know the form of each column of $D V^T$:

\[
[D V^T]_j = 
(\sigma_1 V_{j1}, \sigma_2 V_{j2}, \dots, \sigma_r V_{jr})^T
\]

Now, let us get an expression for the final product of the left side. Using the definition for matrix product we know that each cell is a dot product (we transpose $U$ for getting its i-th row using our column subscript notation):

\[
[U (D V^T)]_{ij} = [U^T]_i \cdot [D V^T]_j
\]

Replacing what we know for the columns of $D V^T$ gives us the following (we dropped the transpose on the right operand, cause it does not matter for the dot product):

\[
[U^T]_i \cdot [D V^T]_j = [U^T]_i \cdot 
(\sigma_1 V_{j1}, \sigma_2 V_{j2}, \dots, \sigma_r V_{jr})
\]

Using the definition of dot product and remembering that $[U^T]_i$ stands for the i-th row of $U$:

\[
[U^T]_i \cdot 
(\sigma_1 V_{j1}, \sigma_2 V_{j2}, \dots, \sigma_r V_{jr}) = 
\sum_{k=1}^r \sigma_k U_{ik} V_{jk}
\]

Thus, we know that each cell of the left hand side looks like this:

\[
[U (D V^T)]_{ij} = \sum_{k=1}^r \sigma_k U_{ik} V_{jk}
\]

If we prove that each cell of the right side of the original equation, also equals this summation, we would be done. Let us see if we can do that. The first thing to realize, is that the double indexing $_{ij}$ of that matrix, gets translated into indexing of each individual rank-1 matrix:

\[
\left[\sum_{k=1}^r \sigma_k u_k v_k^{T}\right]_{ij} =
\sum_{k=1}^r \sigma_k [u_k v_k^{T}]_{ij}
\]

Now, the rank-1 matrix $u_k v_k^{T}$ is formed by all the possible pair-products between the scalar elements in $u_k$ and those in $v_k^T$. Thus, if we take the element $_{ij}$ it will be just a product of two scalars: the i-th element of $u_k$ and the j-th element of $v_k^T$. 

But the i-th element of $u_k$ is the i-th entry of the column $[U]_k$, which is $U_{ik}$. Similarly, the j-th element of $v_k^T$ is the j-th entry of the k-th column in $V$, which is $V_{jk}$ (we can ignore the transpose operator in $v_k^T$, cause it is just to put that column horizontal and allow for the matrix product). Putting this explanation into equations would translate into:

\[
\sum_{k=1}^r \sigma_k [u_k v_k^{T}]_{ij} =
\sum_{k=1}^r \sigma_k U_{ik} V_{jk}
\]

So we can tell now what is the expression for each cell on the right hand side:

\[
\left[\sum_{k=1}^r \sigma_k u_k v_k^{T}\right]_{ij} =
\sum_{k=1}^r \sigma_k U_{ik} V_{jk}
\]

And since cell-wise, both matrices lead to same expression, we can conclude they are the same.
