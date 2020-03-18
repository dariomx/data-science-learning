---
output:
  pdf_document: default
  html_document: default
---

Show that the rows of $A_k$ are the projections of the rows of $A$ onto the subspace of $V_k$ spanned by the first $k$ right singular vectors.

Using the result of the exercise a), and the definition of $A_k$, we know that:

\[
A_k = \sum_{i=1}^k \sigma_i u_i v_i^T
\]

Let us leverage now the result from exercise b):

\[
\sum_{i=1}^k \sigma_i u_i v_i^T = 
\sum_{i=1}^k \sigma_i \left(\frac{1}{\sigma_i} A v_i\right) v_i^T =
\sum_{i=1}^k A v_i v_i^T
\]

We can tell now something about the structure of matrix $A_k$, in terms of $A$:

\[
A_k = \sum_{i=1}^k A v_i v_i^T
\]

Though we would like to go deeper, and tell something about its rows. For that purpose, let us mirror our previous syntax for projecting by column; but now for the case of rows. The expression $[M]^j$ will represent the j-th row of the matrix $M$. Using this notation, and the previous result, let us reveal the structure of a single row of $A_k$:

\[
[A_k]^j = 
\sum_{i=1}^k ([A]^j \cdot v_i) v_i^T
\]

The above result holds, cause the matrix-vector product $A v_i$ would be restricted to j-th row of $A$, instead of doing dot product of $v_i$ against all rows of $A$. 

This result itself suffices, as per definitions, the equation above shows that each row of $[A_k]^j$ can be expressed as the projection of the associated row $[A]_j$, onto the first $k$ right singular vectors. The transpose operator of $v_i^T$ can be ignored, as it merely ensures a proper shape on the result (we want horizontal vectors here).


