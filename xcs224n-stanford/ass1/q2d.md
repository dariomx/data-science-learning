---
output:
  pdf_document: default
  html_document: default
---

Show that 

\[
A_k = \underset{rank(B)=k}{argmin}\,\Vert A - B \Vert_F
\]

It did not occur me how to leverage the hint, though I used a different insight, from one of the lectures of Prof. Strang from MIT. The main idea is that the result would be easier to prove for a diagonal matrix $A$, let us try to argument that first then.

Due definition of Frobenius norm, any element of $B$ which is not under the diagonal will bring an extra error. Any non-diagonal matrix $B$ we can think of, will lose the competition against a derived matrix which only takes its diagonal. This means that, the optimal structure of B is diagonal as well.

Now, what about rank of $B$? Well, given its revealed diagonal structure, we can easily tell what is its rank: the number of non zero entries on its diagonal. This means that, if we want a rank $k$, we need to make $r-k$ entries zero; where $r$ is the rank of $A$.

So far so good, we know that $B$ must be diagonal and that $k$ of its values must be non zero. But what to put there? What is the best diagonal $B$ we can subtract from diagonal $A$, in order to minimize the error? Speaking at the level of each cell in the diagonals, the smallest possible error is zero; and the only way to achieve that is by copying plain values from $A$ into $B$. Which ones to take? Well, if we fail to copy the biggest values, they will punish us more on the error; therefore, we better take the biggest $k$ values from diagonal $A$ and ensure they get copied into corresponding places at $B$. Let us see what would be the remaining error, assuming $A \in \mathbb{R}^{r \times r}$ and also that the diagonal $A$ has its values sorted descendently:

\[
error = \sum_{i=1}^r (A_{ii} - B_{ii})^2 
= \sum_{i=k+1}^r (A_{ii} - B_{ii})^2 
= \sum_{i=k+1}^r A_{ii}
\]

It is interesting to realize that error above no longer depends on matrix $B$, but is rather intrinsic to the problem (matrix $A$). Any other matrix $B$ which is not diagonal, or does not take the biggest $k$ entries of diagonal $A$, will produce a bigger error. So looks like we are done for the diagonal case.

And you might wonder, "well ... that is cheating because diagonal matrices are easy, they are special". But as Prof. Strang explains, diagonal matrices are not that special at all. This is the beauty of the SVD factorization: it tells us that, no matter what matrix we have, it can be thought as a diagonal matrix $D$. In other words, we can always factorize $A$ as:

\[
A = U D V^T
\]

which can be re-arranged as:

\[
U^T A V = D
\]

and above can be interpreted as this: no matter what transformation $A$ does to space, there are always a couple of changes of basis ($V$ and $U^T$), which can re-express such transformation as a simple diagonal one. That is, all matrices are in essence a series of compression/expansions along some special axis (basis); and it is much easier to understand that transformation than an arbitrary matrix $A$. 

Note: we exclude reflections along axis in above description of the diagonal, due the nature of singular values (which are always non negative).

Alright, then, if any matrix is in essence a diagonal with sorted and positive values; then what we reasoned before for diagonals applies to any matrix and we have finished the (informal) proof. Hopefully this will suffice to get those extra points.


