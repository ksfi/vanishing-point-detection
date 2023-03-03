# vanishing-point-detection

Vanishing-point detection using the Hough space

## Hough-space

The Hough Transform for straight lines detection consists in mapping the image plane into a parameters plane, a straight line is thus described by a parametric representation
$$\rho = x\cos \theta + y \sin \theta \iff y = \text{m} x + \text{q} $$ 

So, to each point $(x_i, y_i)$ is associated a curve $\rho(\theta, x_i, y_i)$ in the plane $(\rho, \theta)$.

Two points in the image plane that belong to the same line thus determine a point in the polar plane $(\rho_i, \theta_i)$ given by the intersection if two sine curves.

<img width="758" alt="Capture d’écran 2023-03-03 à 19 20 17" src="https://user-images.githubusercontent.com/126407732/222797616-ed9a4317-c82e-42f1-bc52-7c32975f0d9a.png">
