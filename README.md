# Learning-3D-Vision-with-Inverse-Graphics---Part-II

## Plan of Action
1. [Scene Representations](#sr)
2. [The Rendering Equation](#tdr)
3. [Neural Surface Rendering](#nsr)
4. [Differentiable Rendering](#dr)
5. [Neural Volume Rendering](#nvr)


---------------------
<a name="sr"></a>
## 1. Scene Representations
In Part I of the series, we saw how we can parametrize a surface with mesh, point cloud or voxel grid. Now, we want to parameterize 'everything, everywhere, all at once'. How do we do that? While surface representations rely on surface parameterization, volumetric representations require volumetric parameterization—more specifically called '**field parametrization**'. To explain this change in terminology, let's define what a field is:

_In physics, functions that output values for space-time input coordinates are referred to as fields._

With volumetric representation, we want to create these representations for every point in 3D space(not just on surfaces). That is, we densely map every coordinate in space to the properties of that 3D coordinate. The term 'field', therefore, represents any function that takes a space, time, or space-time coordinate as input and maps it to a known quantity at that coordinate. 

We refer to fields that are fully or partially parametrized by neural networks as neural fields. 

We will build neural fields that will take the ```x,y``` pixel coordinats of an image and map it to its RGB value. We will also build neural fields that will take the ```x,y,z``` coordinates of a 3D space and map it to a probability value that indicates the likelihood of being occupied.

- **2D Images:** $f: \mathbb{R}^2 \to \mathbb{R}^3$, where outputs are RGB colors
- **3D Occupancy:** $f: \mathbb{R}^3 \to \mathbb{R}$, where outputs represent "density"



**1. Occupancy Field**

Once way if to use an **occupancy field**. It is a **continuous function** that maps any 3D point to a **probability value**([0, 1]) that indicates the **likelihood** of being occupied. We can choose the level set to be ```0.5``` such that above that value it is **classifed** as "occupied" and below is "not occupied" - similar to how a **logistic regression** would work in 3D space.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5866832e-102b-4a8e-953f-f10109532135" width="30%" />
</p>

Occupanyc fields are **implicit representations** as we do not store actual surface or volume points. Instead, we store a function (a neural network) that can tell us for any point whether it's occupied. The surface is defined **implicitly** as the level set where ```f(x,y,z) = 0.5```.

```python
# Implicit (Occupancy Field)
def occupancy_field(x, y, z):
    # Implicitly defines shape through a function
    return neural_network(torch.tensor([x, y, z]))  # Returns if point is inside/outside
```

**2. Signed Distance Field**

Another example is to use a SDF where for every point in 3D space, it gives us the distance to te surface. Normally, negative values = inside, positive values = outside and zero = exactly on surface.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d73fcc83-f4cc-4a8d-85a7-23441bae311c" width="30%" />
</p>

The SDF is an **implicit continuous function** that maps any 3D point to a **distance value**. We can choose the level set to be ```0``` such that above that value(+ve values) is **classified** as "outside" and below(-ve values) is "inside". 


```python
# Implicit (SDF)
def signed_distance_field(x, y, z):
    # Returns actual distance (negative inside, positive outside)
    return network(torch.tensor([x, y, z]))
```

-----------

**3. Neural Fields**

SDF can represent non-complex functions such as a sphere or a cube. However, for more complex shapes, we need more complex functions. This is where **neural fields** come into play. NN are known as universal approximators, meaning they can closely  represent any continuous function given enough parameters.

SDF and Occupancy fields can be represented using a **neural field** which is a **continuous function** that maps any 3D point to a **distance value** or **probability value**.


```python
def neural_field_sdf(x,y,z):
    coords = torch.tensor([x,y,z])
    return network(coords)  # Can output distance (SDF)

def neural_field_occupancy(x,y,z):
    coords = torch.tensor([x,y,z])
    return network(coords)  # Can output occupancy [0,1]
```

Neural fields are compact multi-resolution representations of shapes that is also continuous. To train a neural field, we can sample 3D cooridnates from the ground-truth and then feed it to the NN to reproduce the occupancy function. We then extract the surface with marching cubes algo.

In summary:

- Storage memory does not grow with the complexity of the shape.(spatial resolution or number of spatial dimensions)
- GPU-memory intensive: Each query requires a forward pass through the NN.
- Over-optimization techniqe with adaptive sampling in detailed regions: Network can learn to allocate parameters efficiently.


-------------

### 1.1 Neural Field: MLP
The simplest neural field architectures we can implement is an MLP. Our neural network has 2 inputs: ```x,y``` and 3 outputs: ```r,g,b```, with 2 hidden layers of 128 neurons each.

We first implement the MLP without any positional encoding. Positional encoding is a technique used to encode the input coordinates into a higher-dimensional space, allowing the network to learn more complex and structured representations. I explained more about it in my project: [Assessing NeRF's Efficacy in 3D Model Reconstruction: A Comparative Analysis with Blender](https://github.com/yudhisteer/Assessing-NeRF-s-Efficacy-in-3D-Model-Reconstruction-A-Comparative-Analysis-with-Blender).


We observe that the MLP **without positional encoding** outputs a blurry image. This is because the network is not able to learn the spatial structure of the image. On the other hand, the MLP **with positional encoding** is able to learn the high frequency details of the image better.


<div style="text-align: center;">
<table>
  <tr>
    <th>MLP without Positional Encoding = 0.0104</th>
    <th>MLP with Positional Encoding = 0.0045</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/3162aae9-859d-48d4-946b-bf74742da72d" alt="MLP without Positional Encoding" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/c7aa2d9a-2044-43a2-96dc-0b6bbde9c4fb" alt="MLP with Positional Encoding" width="300"></td>
  </tr>
</table>
</div>

**Neural Tangent Kernel**

Now we may be under the assumption that we are doing some "Artificial Intelligence" here but actually we are not. We are just doing "Bi-linear interpolation ++".

The role of our MLP and that in a NeRF is the role of an **interpolation kernel**.  Neural implicit representations essentially perform **kernel regression/interpolation** based on the **Neural Tangent Kernel**. Te network just interpolates between training points.

Given a training set {(**x<sub>i</sub>**, **y<sub>i</sub>**)}<sub>i</sub>, a kernel function makes predictions on a point **x** by interpolating labels **y<sub>i</sub>** in the training set according to pairwise weights between **x<sub>i</sub>** to **x** as meaured by the kernel function **k**.

$$
f(\textbf{x}) = \sum_{i=1}^{n}(\textbf{K}^{-1}\textbf{y})_ik(\textbf{x}_i, \textbf{x})
$$


$$
k_{NTK}(x_1, x_2) =  \frac{\partial f(x_1; \theta)}{\partial \theta} \cdot \frac{\partial f(x_2; \theta)}{\partial \theta}
$$

where **K**<sub>ij</sub> = k<sub>NTK</sub>(**x<sub>i</sub>**, **x<sub>j</sub>**), k is the kernel function with k >= 0.

```python
def ntk_inference(query_point, training_points, training_values, ntk_kernel):
   # Initialize list to store kernel values
   k_values = []
   
   # Compute kernel between query point and each training point
   for train_point in training_points:
       k = ntk_kernel(query_point, train_point)
       k_values.append(k)
   
   # Initialize prediction
   prediction = 0.0
   
   # Compute weighted sum
   for i in range(len(training_points)):
       prediction += k_values[i] * training_values[i]
   
   return prediction
```

For an input coordinate ```x``` that is close to a training point ```x'```, former has to be very close to the latter in order to produce the correct output. Hence, a blurry image shows that the neighboring pixels are coupled together - the one with ReLU activation functions.




### 1.2 Siren

SIREN: loss: 0.0070

![ezgif com-animated-gif-maker (5)](https://github.com/user-attachments/assets/831936cb-612e-441e-8e5d-02813cbc22ec)


### 1.3 Grid

```python
    def query(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Sample the image at the specified coordinates."""
        # Convert from [0, 1] to [-1, 1] range
        grid = 2.0 * coordinates - 1.0 # shape: [N, 2]
        
        # Reshape coordinates to match grid_sample expected format
        grid = grid.view(coordinates.size(0), 1, 1, 2) # shape: [N, 1, 1, 2]
        
        # Expand image for batch
        batch_size = coordinates.size(0)
        image = self.image.unsqueeze(0).expand(batch_size, -1, -1, -1) # shape: [N, 3, 512, 512]
        
        # Sample from image
        sampled = F.grid_sample(
            image, 
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ) # shape: [N, 3, 1, 1]
        
        return sampled.squeeze(-1).squeeze(-1) # shape: [N, 3]
```

![image](https://github.com/user-attachments/assets/aee2d95c-f8c4-40bf-9241-335904a5df09)


Grid: loss: 0.0060

![ezgif com-animated-gif-maker (4)](https://github.com/user-attachments/assets/a2af3637-fa8c-413e-9687-36ceb4b4e44c)

### 1.4 Hybrid
Hybrid Grid: loss: 0.0021

![ezgif com-animated-gif-maker (3)](https://github.com/user-attachments/assets/d0e90d92-fffc-4eee-897a-0fb848797766)


### 1.5 Ground Plan

loss: loss: 0.0001

![ezgif com-animated-gif-maker (6)](https://github.com/user-attachments/assets/2ba5ec07-c2ac-45a1-9b12-cb67ed61346d)


Comparison:

![comparison](https://github.com/user-attachments/assets/648e7d6f-b183-41e5-8eeb-450807ff5394)


---------------------
<a name="tdr"></a>
## 2. The Rendering Equation


---------------------

## References
1. https://github.com/learning3d/assignment3
2. https://andrewkchan.dev/posts/lit-splat.html
3. https://learning3d.github.io/schedule.html
4. https://www.scenerepresentations.org/courses/inverse-graphics-23/
5. https://www.songho.ca/opengl/gl_transform.html
6. https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
7. https://towardsdatascience.com/differentiable-rendering-d00a4b0f14be
8. https://blog.qarnot.com/article/an-overview-of-differentiable-rendering
