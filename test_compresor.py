import taichi as ti
import numpy as np
import time

# ════════════════════════════════════════════════════════════════════
# WAG ULTIMATE: ENTROPIC POTENTIAL VERSION
# Potencial emerge de entropía: V_eff = T·log(ρ + ε)
# Física: Schrödinger logarítmica → dark energy emergente
# ════════════════════════════════════════════════════════════════════

try:
    ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_GB=4)
    print("✓ Inicializado en CUDA")
except:
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    print("⚠ Fallback a CPU")

# ────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ────────────────────────────────────────────────────────────────────

RESOLUTION = 128      # Voxels (128³ para 60fps estable)
WINDOW_RES = 800      # Ventana de visualización
DT = 0.0005           # Timestep estable
ALPHA = 0.5           # Dispersión (difusión cuántica)

# 🔥 NUEVOS PARÁMETROS ENTRÓPICOS
T_ENT = 0.15          # Temperatura entrópica efectiva
EPS = 1e-6            # Epsilon para estabilidad del log

# ────────────────────────────────────────────────────────────────────
# CAMPOS TAICHI
# ────────────────────────────────────────────────────────────────────

psi_real = ti.field(dtype=ti.f32)
psi_imag = ti.field(dtype=ti.f32)
psi_real_new = ti.field(dtype=ti.f32)
psi_imag_new = ti.field(dtype=ti.f32)

# Sparse Voxel Octree (Optimización de memoria)
block = ti.root.pointer(ti.ijk, RESOLUTION // 8)
pixel = block.dense(ti.ijk, 8)
pixel.place(psi_real, psi_imag, psi_real_new, psi_imag_new)

# Campos auxiliares
density = ti.field(dtype=ti.f32, shape=(RESOLUTION, RESOLUTION, RESOLUTION))
phase_field = ti.field(dtype=ti.f32, shape=(RESOLUTION, RESOLUTION, RESOLUTION))

# Framebuffer para raymarching
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WINDOW_RES, WINDOW_RES))

# ────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN: Red Cósmica
# ────────────────────────────────────────────────────────────────────

@ti.kernel
def initialize_cosmic_web(num_galaxies: ti.i32, scale: ti.f32):
    """
    Inicializa galaxias como solitones dispersos.
    Paralelizado por voxel (GPU-friendly).
    """
    for i, j, k in ti.ndrange(RESOLUTION, RESOLUTION, RESOLUTION):
        total_real = 0.0
        total_imag = 0.0
        
        for gid in range(num_galaxies):
            seed = gid * 7919
            
            # Hash determinista para posición y propiedades
            h1 = (seed * 1103515245 + 12345) % 2147483647
            h2 = (seed * 1664525 + 1013904223) % 2147483647
            h3 = (seed * 22695477 + 1) % 2147483647
            
            cx = ti.f32(h1 % RESOLUTION)
            cy = ti.f32(h2 % RESOLUTION)
            cz = ti.f32(h3 % RESOLUTION)
            
            width = scale * (1.0 + ti.f32((seed % 100) / 200.0))
            
            dx = ti.f32(i) - cx
            dy = ti.f32(j) - cy
            dz = ti.f32(k) - cz
            r_sq = dx*dx + dy*dy + dz*dz
            
            # Cutoff para eficiencia
            if r_sq < (width * 5.0)**2:
                amp = 0.5 + ti.f32((seed % 50) / 100.0)
                phase_off = ti.f32((seed % 628) / 100.0)
                
                profile = amp * ti.exp(-r_sq / (2.0 * width * width))
                phase = phase_off + ti.atan2(dy, dx) * 2.0
                
                total_real += profile * ti.cos(phase)
                total_imag += profile * ti.sin(phase)
        
        psi_real[i, j, k] = total_real
        psi_imag[i, j, k] = total_imag

# ────────────────────────────────────────────────────────────────────
# FÍSICA: Split-Step con POTENCIAL ENTRÓPICO
# ────────────────────────────────────────────────────────────────────

@ti.func
def laplacian(fr, fi, i, j, k):
    """Laplaciano 3D con condiciones periódicas"""
    ip = (i + 1) % RESOLUTION
    im = (i - 1 + RESOLUTION) % RESOLUTION
    jp = (j + 1) % RESOLUTION
    jm = (j - 1 + RESOLUTION) % RESOLUTION
    kp = (k + 1) % RESOLUTION
    km = (k - 1 + RESOLUTION) % RESOLUTION
    
    lr = (fr[ip,j,k] + fr[im,j,k] + fr[i,jp,k] + 
          fr[i,jm,k] + fr[i,j,kp] + fr[i,j,km] - 6.0*fr[i,j,k])
    li = (fi[ip,j,k] + fi[im,j,k] + fi[i,jp,k] + 
          fi[i,jm,k] + fi[i,j,kp] + fi[i,j,km] - 6.0*fi[i,j,k])
    return lr, li

@ti.kernel
def step1(dt: ti.f32, alpha: ti.f32):
    """Paso 1: Evolución dispersiva (parte lineal)"""
    for i, j, k in ti.ndrange(RESOLUTION, RESOLUTION, RESOLUTION):
        r, im = psi_real[i,j,k], psi_imag[i,j,k]
        lr, li = laplacian(psi_real, psi_imag, i, j, k)
        theta = alpha * dt * (lr + li)
        c, s = ti.cos(theta), ti.sin(theta)
        psi_real_new[i,j,k] = r*c - im*s
        psi_imag_new[i,j,k] = r*s + im*c

@ti.kernel
def step2(dt: ti.f32, T: ti.f32):
    """
    🔥 PASO 2: POTENCIAL ENTRÓPICO
    
    V_ent = T·log(ρ + ε)
    
    donde ρ = |ψ|² es la densidad local.
    
    Esto emerge de δ(T·S)/δρ con S = -ρ log ρ (entropía de Shannon).
    
    Efectos físicos:
    - Regiones densas: presión entrópica (anti-colapso)
    - Regiones vacías: expansión tipo dark energy
    - No blowup (log suaviza vs. potencias)
    """
    for i, j, k in ti.ndrange(RESOLUTION, RESOLUTION, RESOLUTION):
        r = psi_real_new[i,j,k]
        im = psi_imag_new[i,j,k]

        rho = r*r + im*im

        # 💥 POTENCIAL ENTRÓPICO (aquí está la magia)
        V_ent = T * ti.log(rho + EPS)

        theta = V_ent * dt
        c, s = ti.cos(theta), ti.sin(theta)

        psi_real[i,j,k] = r*c - im*s
        psi_imag[i,j,k] = r*s + im*c

        # Actualizar campos visuales
        density[i,j,k] = rho
        phase_field[i,j,k] = ti.atan2(im, r)

@ti.kernel
def swap():
    """Intercambia buffers"""
    for i, j, k in ti.ndrange(RESOLUTION, RESOLUTION, RESOLUTION):
        psi_real[i,j,k] = psi_real_new[i,j,k]
        psi_imag[i,j,k] = psi_imag_new[i,j,k]

def evolve():
    """Evolución split-step completa"""
    step1(DT/2, ALPHA)
    step2(DT, T_ENT)
    swap()
    step1(DT/2, ALPHA)
    swap()

# ────────────────────────────────────────────────────────────────────
# RAYMARCHING: Decodificador Holográfico
# ────────────────────────────────────────────────────────────────────

@ti.func
def sample_density(pos: ti.math.vec3) -> ti.f32:
    """Sampleo trilinear de densidad"""
    x = ti.max(0.0, ti.min(RESOLUTION - 1.001, pos.x))
    y = ti.max(0.0, ti.min(RESOLUTION - 1.001, pos.y))
    z = ti.max(0.0, ti.min(RESOLUTION - 1.001, pos.z))
    
    ix, iy, iz = ti.i32(x), ti.i32(y), ti.i32(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    
    ixp = ti.min(ix + 1, RESOLUTION - 1)
    iyp = ti.min(iy + 1, RESOLUTION - 1)
    izp = ti.min(iz + 1, RESOLUTION - 1)
    
    c000 = density[ix, iy, iz]
    c100 = density[ixp, iy, iz]
    c010 = density[ix, iyp, iz]
    c110 = density[ixp, iyp, iz]
    c001 = density[ix, iy, izp]
    c101 = density[ixp, iy, izp]
    c011 = density[ix, iyp, izp]
    c111 = density[ixp, iyp, izp]
    
    c00 = c000*(1-fx) + c100*fx
    c01 = c001*(1-fx) + c101*fx
    c10 = c010*(1-fx) + c110*fx
    c11 = c011*(1-fx) + c111*fx
    
    c0 = c00*(1-fy) + c10*fy
    c1 = c01*(1-fy) + c11*fy
    
    return c0*(1-fz) + c1*fz

@ti.func
def sample_phase(pos: ti.math.vec3) -> ti.f32:
    """Sampleo de fase para coloración"""
    ix = ti.i32(ti.max(0, ti.min(RESOLUTION-1, pos.x)))
    iy = ti.i32(ti.max(0, ti.min(RESOLUTION-1, pos.y)))
    iz = ti.i32(ti.max(0, ti.min(RESOLUTION-1, pos.z)))
    return phase_field[ix, iy, iz]

@ti.kernel
def raymarch(time: ti.f32, rot_x: ti.f32, rot_y: ti.f32):
    """Raymarching volumétrico con colores de fase"""
    center = ti.math.vec3(RESOLUTION/2, RESOLUTION/2, RESOLUTION/2)
    
    for px, py in pixels:
        u = (ti.f32(px) / WINDOW_RES - 0.5) * 2.0
        v = (ti.f32(py) / WINDOW_RES - 0.5) * 2.0
        
        # Cámara orbital
        angle = rot_y + time * 0.15
        dist = RESOLUTION * 2.0
        cam_pos = ti.math.vec3(
            center.x + ti.cos(angle) * dist,
            center.y + rot_x * RESOLUTION * 0.3,
            center.z + ti.sin(angle) * dist
        )
        
        forward = (center - cam_pos).normalized()
        right = ti.math.vec3(0, 1, 0).cross(forward).normalized()
        up = forward.cross(right)
        
        ray_dir = (forward + u * right + v * up).normalized()
        
        # Raymarch
        color = ti.math.vec3(0.0)
        transmittance = 1.0
        
        num_steps = 200
        t_max = RESOLUTION * 3.0
        dt_ray = t_max / num_steps
        t = 0.0
        
        for _ in range(num_steps):
            pos = cam_pos + ray_dir * t
            
            if (0 <= pos.x < RESOLUTION and 
                0 <= pos.y < RESOLUTION and 
                0 <= pos.z < RESOLUTION):
                
                rho = sample_density(pos)
                
                if rho > 0.005:
                    sigma = rho * 6.0
                    
                    ph = sample_phase(pos)
                    heat = ti.min(rho * 8.0, 1.0)
                    
                    # RGB de fase (rueda de color)
                    r_val = 0.5 + 0.5 * ti.cos(ph)
                    g_val = 0.5 + 0.5 * ti.cos(ph + 2.094)
                    b_val = 0.5 + 0.5 * ti.cos(ph + 4.189)
                    
                    emission = ti.math.vec3(r_val, g_val, b_val) * heat
                    
                    alpha_val = 1.0 - ti.exp(-sigma * dt_ray)
                    color += transmittance * emission * alpha_val
                    transmittance *= (1.0 - alpha_val)
                    
                    if transmittance < 0.01:
                        break
            
            t += dt_ray
        
        bg = ti.math.vec3(0.05, 0.05, 0.12) * (1.0 + v * 0.2)
        pixels[px, py] = color + transmittance * bg

# ────────────────────────────────────────────────────────────────────
# ANÁLISIS: Compresión Espectral
# ────────────────────────────────────────────────────────────────────

@ti.kernel
def measure_norm() -> ti.f32:
    """Mide la norma L2 del campo"""
    total = 0.0
    for i, j, k in ti.ndrange(RESOLUTION, RESOLUTION, RESOLUTION):
        r = psi_real[i, j, k]
        im = psi_imag[i, j, k]
        total += r*r + im*im
    return ti.sqrt(total)

@ti.kernel
def count_active_voxels(threshold: ti.f32) -> ti.i32:
    """Cuenta voxels con densidad > threshold"""
    count = 0
    for i, j, k in ti.ndrange(RESOLUTION, RESOLUTION, RESOLUTION):
        r = psi_real[i, j, k]
        im = psi_imag[i, j, k]
        if r*r + im*im > threshold:
            count += 1
    return count

def analyze_compression():
    """Análisis FFT de compresión espectral"""
    field = psi_real.to_numpy() + 1j * psi_imag.to_numpy()
    
    print("\n" + "="*60)
    print("COMPRESIÓN DIMENSIONAL SEMÁNTICA")
    print("="*60)
    
    original_size = field.nbytes / (1024**2)
    print(f"Campo: {RESOLUTION}³ = {RESOLUTION**3:,} voxels")
    print(f"Memoria: {original_size:.2f} MB")
    
    print("Computando FFT 3D...")
    fft_field = np.fft.fftn(field)
    power = np.abs(fft_field)**2
    
    flat_power = power.flatten()
    sorted_idx = np.argsort(flat_power)[::-1]
    
    total_power = np.sum(flat_power)
    cumsum = np.cumsum(flat_power[sorted_idx])
    
    n_99 = np.searchsorted(cumsum, 0.99 * total_power)
    n_95 = np.searchsorted(cumsum, 0.95 * total_power)
    
    total_modes = RESOLUTION**3
    ratio = total_modes / n_99 if n_99 > 0 else 1.0
    
    print(f"\n95% energía: {n_95:,} modos ({n_95/total_modes*100:.3f}%)")
    print(f"99% energía: {n_99:,} modos ({n_99/total_modes*100:.3f}%)")
    print(f"Compresión: {ratio:.1f}x")
    if ratio > 0:
        print(f"Memoria comprimida: {original_size/ratio:.2f} MB")
    print("="*60 + "\n")
    
    return ratio

# ────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("WAG ULTIMATE - ENTROPIC POTENTIAL VERSION")
    print("Física: V_eff = T·log(ρ + ε)")
    print("="*60)
    print("Controles:")
    print("  Mouse izquierdo + arrastrar - Rotar cámara")
    print("  SPACE - Analizar compresión FFT")
    print("  R - Reiniciar universo")
    print("  1-5 - Cambiar número de galaxias")
    print("  P - Pausar/Reanudar")
    print("  Q/ESC - Salir")
    print("="*60)
    
    num_galaxies = 30
    print(f"\nInicializando {num_galaxies} galaxias...")
    initialize_cosmic_web(num_galaxies, 8.0)
    
    initial_norm = measure_norm()
    active = count_active_voxels(0.01)
    sparsity = 100 - (active / RESOLUTION**3 * 100)
    
    print(f"✓ Norma: {initial_norm:.2f}")
    print(f"✓ Dispersión: {sparsity:.1f}% vacío")
    print(f"✓ Voxels activos: {active:,}")
    
    gui = ti.GUI("WAG - Entropic Potential", 
                 res=(WINDOW_RES, WINDOW_RES))
    
    rot_x, rot_y = 0.0, 0.0
    last_mx, last_my = 0.5, 0.5
    step = 0
    paused = False
    
    while gui.running:
        mx, my = gui.get_cursor_pos()
        
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in ['q', ti.GUI.ESCAPE]:
                break
            elif gui.event.key == ' ':
                print("\n[Analizando compresión...]")
                analyze_compression()
            elif gui.event.key == 'r':
                print("\n♻ Reiniciando...")
                initialize_cosmic_web(num_galaxies, 8.0)
                step = 0
            elif gui.event.key == 'p':
                paused = not paused
            elif gui.event.key == '1':
                num_galaxies = 10
                initialize_cosmic_web(num_galaxies, 8.0)
                step = 0
            elif gui.event.key == '2':
                num_galaxies = 30
                initialize_cosmic_web(num_galaxies, 8.0)
                step = 0
            elif gui.event.key == '3':
                num_galaxies = 50
                initialize_cosmic_web(num_galaxies, 8.0)
                step = 0
            elif gui.event.key == '4':
                num_galaxies = 100
                initialize_cosmic_web(num_galaxies, 8.0)
                step = 0
            elif gui.event.key == '5':
                num_galaxies = 200
                initialize_cosmic_web(num_galaxies, 8.0)
                step = 0
        
        if gui.is_pressed(ti.GUI.LMB):
            rot_y += (mx - last_mx) * 8.0
            rot_x += (my - last_my) * 8.0
            rot_x = max(-2.0, min(2.0, rot_x))
        
        last_mx, last_my = mx, my
        
        if not paused:
            for _ in range(3):
                evolve()
            step += 3
        
        raymarch(step * DT, rot_x, rot_y)
        gui.set_image(pixels)
        
        norm = measure_norm()
        if initial_norm > 0:
            deviation = abs(norm - initial_norm) / initial_norm * 100
        else:
            deviation = 0.0
        
        gui.text(f"WAG ENTROPIC - {num_galaxies} Galaxies", 
                 pos=(0.05, 0.95), color=0x00FFFF, font_size=20)
        gui.text(f"Step: {step} | Norm: {norm:.2f} ({deviation:.2f}%)", 
                 pos=(0.05, 0.05), color=0x00FF00, font_size=16)
        gui.text(f"T_ent={T_ENT:.2f} | Sparsity: {sparsity:.1f}%", 
                 pos=(0.05, 0.02), color=0xFFFF00, font_size=14)
        
        if paused:
            gui.text("PAUSED", pos=(0.45, 0.5), 
                     color=0xFF0000, font_size=30)
        
        gui.show()
    
    print("\n✓ Simulación finalizada")

if __name__ == "__main__":
    main()
