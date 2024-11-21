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


### 1.1 MLP

![image](https://github.com/user-attachments/assets/aee2d95c-f8c4-40bf-9241-335904a5df09)

MLP w/o: loss: 0.0104

![ezgif com-animated-gif-maker](https://github.com/user-attachments/assets/3162aae9-859d-48d4-946b-bf74742da72d)


MLP with: loss: 0.0045

![ezgif com-animated-gif-maker (1)](https://github.com/user-attachments/assets/c7aa2d9a-2044-43a2-96dc-0b6bbde9c4fb)




### 1.2 Siren

SIREN: loss: 0.0070

![ezgif com-animated-gif-maker (5)](https://github.com/user-attachments/assets/831936cb-612e-441e-8e5d-02813cbc22ec)


### 1.3 Grid
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
