# S2IR

Scale Invariant Image Refiner

Primarily uses relative positional coordinates (fourier series) added to the pixel "tokens". Then uses axial attention
and text conditioning with an FFN in style of text transformer blocks, but adapted for images. Trained to predict
epsilon noise to be used as a diffusion model. Technically doesn't care about render resolution if trained properly.