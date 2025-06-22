import numpy as np

class TeaCache:
    def __init__(self, rel_l1_thresh=0.6,
                  coefficients=[4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
                  warm_up_steps=2, num_steps=None, forward_func=None):
        self.accumulated_rel_l1_distance = 0
        self.previous_latent_input = None
        self.previous_residual = None
        self.skipped_steps = 0
        self.num_steps = num_steps
        self.forward_func = forward_func
        self.rel_l1_thresh = rel_l1_thresh
        self.coefficients = coefficients
        self.warm_up_steps = warm_up_steps
        self.rescale_func = np.poly1d(coefficients)
        
        
    def should_calculate(self, current_step, x_t):
        """Determine if calculation should be performed or cached result can be used"""
        # Always calculate for first few steps and last step
        if current_step < self.warm_up_steps or current_step == self.num_steps-1:
            self.accumulated_rel_l1_distance = 0
            return True
            
        if self.previous_latent_input is not None:
            # Calculate relative L1 distance
            rel_l1 = (x_t - self.previous_latent_input).abs().mean() / (self.previous_latent_input.abs().mean() + 1e-8)
            rel_l1_value = rel_l1.item()
            
            # Apply polynomial rescaling
            self.accumulated_rel_l1_distance += self.rescale_func(rel_l1_value)
            
            # Determine if calculation can be skipped
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                self.skipped_steps += 1
                return False
            else:
                self.accumulated_rel_l1_distance = 0
                return True
        
        return True
        
    def update_cache(self, x_t, v_t):
        """Update cache with new values"""
        self.previous_latent_input = x_t
        self.previous_residual = v_t
        
    def get_cached_residual(self):
        """Get cached residual value"""
        return self.previous_residual
        
    def reset(self):
        """Reset cache state"""
        self.accumulated_rel_l1_distance = 0
        self.previous_latent_input = None
        self.previous_residual = None
        self.skipped_steps = 0
    
    def process(self, current_step, x_t, **kwargs):
        """Process the input, either performing the calculation or returning cached result
        
        Args:
            current_step: Current diffusion step
            x_t: Current latent input
            **kwargs: Additional arguments to pass to the forward_func
            
        Returns:
            The residual calculation result (from forward_func or cache)
        """
        if self.forward_func is None:
            raise ValueError("forward_func must be set before using process()")
            
        # Check if we should calculate or use cache
        should_calc = self.should_calculate(current_step, x_t)
        
        if not should_calc:
            # Use cached result
            return self.get_cached_residual()
        
        # Perform calculation using the provided function
        v_t = self.forward_func(x_t=x_t, **kwargs)
        
        # Update cache with new values
        self.update_cache(x_t, v_t)
        
        return v_t
