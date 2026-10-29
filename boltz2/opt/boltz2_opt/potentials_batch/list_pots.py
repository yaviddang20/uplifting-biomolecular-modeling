from boltz.model.potentials.potentials import get_potentials
sa = {"fk_steering": True, "physical_guidance_update": True,
      "contact_guidance_update": True, "num_particles": 3, "num_gd_steps": 20,
      "fk_lambda": 4.0, "fk_resampling_interval": 3, "guidance_update": True,
      "max_fk_noise_level": 5.0, "contact_max_distance": 6.0}
pots = get_potentials(sa, boltz2=True)
print("active potentials under --use_potentials:")
for p in pots:
    print("  -", type(p).__name__)
