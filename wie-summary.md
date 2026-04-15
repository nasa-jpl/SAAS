Below is a structured summary of the specific equations from Section II of Wie et al. (1989), formatted specifically for an LLM or developer to implement a Quaternion Feedback Regulator for spacecraft eigenaxis rotations.Implementation Context

System: Rigid spacecraft.
Goal: Large-angle, rest-to-rest maneuver about the Euler eigenaxis (shortest angular path).
Inputs: Current angular velocity ω\omega, current attitude quaternion qq.
Outputs: Control torque vector uu.


1. System Dynamics (Plant Model)
These equations define the physical behavior of the spacecraft that the controller must manage.
Euler's Equations of Motion:
Jω˙=−ω×Jω+uJ\dot{\omega} = -\omega^\times J\omega + u
Where:

JJ: 3×33 \times 3 Inertia Matrix (symmetric, positive definite).
ω=[ω1,ω2,ω3]T\omega = [\omega_1, \omega_2, \omega_3]^T: Angular velocity vector in body frame.
u=[u1,u2,u3]Tu = [u_1, u_2, u_3]^T: Control torque vector.
ω×\omega^\times: Skew-symmetric matrix of ω\omega (defined below).

Skew-Symmetric Matrix Definition:
ω×=[0−ω3ω2ω30−ω1−ω2ω10]\omega^\times = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}
2. Quaternion Kinematics
These equations describe how the attitude evolves over time based on angular velocity.
Quaternion Vector:
q=[q1,q2,q3,q4]Tq = [q_1, q_2, q_3, q_4]^T

Vector part: qv=[q1,q2,q3]Tq_v = [q_1, q_2, q_3]^T (Direction of Euler axis).
Scalar part: q4q_4 (Related to rotation angle ϕ\phi).

Kinematic Differential Equation:
q˙=12Ω(ω)q\dot{q} = \frac{1}{2} \Omega(\omega) q
Where Ω(ω)\Omega(\omega) is the 4×44 \times 4 matrix:
Ω(ω)=[0ω3−ω2ω1−ω30ω1ω2ω2−ω10ω3−ω1−ω2−ω30]\Omega(\omega) = \begin{bmatrix} 0 & \omega_3 & -\omega_2 & \omega_1 \\ -\omega_3 & 0 & \omega_1 & \omega_2 \\ \omega_2 & -\omega_1 & 0 & \omega_3 \\ -\omega_1 & -\omega_2 & -\omega_3 & 0 \end{bmatrix}
Alternative compact form often used in code:
q˙v=12(q4I3×3+qv×)ω\dot{q}_v = \frac{1}{2} (q_4 I_{3\times3} + q_v^\times) \omega
q˙4=−12qvTω\dot{q}_4 = -\frac{1}{2} q_v^T \omega
Unit Norm Constraint:
q12+q22+q32+q42=1q_1^2 + q_2^2 + q_3^2 + q_4^2 = 1
3. Error Quaternion Calculation
The controller operates on the error between the current attitude and the desired attitude.
Error Quaternion Definition:
qe=qc−1⊗qq_e = q_c^{-1} \otimes q
(Note: In the paper, for regulation to the reference frame where qc=[0,0,0,1]q_c = [0,0,0,1], qeq_e simplifies to the current qq.)
Quaternion Multiplication Rule (p⊗rp \otimes r):
If qe=[e1,e2,e3,e4]Tq_e = [e_1, e_2, e_3, e_4]^T, and assuming qc=[0,0,0,1]q_c = [0,0,0,1] (regulation case):
ev=qve_v = q_v
e4=q4e_4 = q_4
Small Angle Approximation (Optional, for initialization checks):
If qc=[0,0,0,1]q_c = [0,0,0,1] and angles are small:
ev≈12θe_v \approx \frac{1}{2} \theta
(where θ\theta are Euler angles).
4. The Quaternion Feedback Regulator (Control Law)
This is the core algorithm derived in Section II (Eq. 9).
General Control Law:
u=−ω×Jω−Dω−Kqeu = -\omega^\times J \omega - D \omega - K q_e
Where:

−ω×Jω-\omega^\times J \omega: Gyroscopic Decoupling Term. Counteracts natural gyroscopic coupling.

Implementation Note: The paper notes this term can be omitted (μ=0\mu=0) for slow maneuvers or if robustness is prioritized over strict eigenaxis tracking, but for the ideal eigenaxis case, it is included.


DD: 3×33 \times 3 Damping Gain Matrix.
KK: 3×33 \times 3 Quaternion Feedback Gain Matrix.
qeq_e: Vector part of the error quaternion [e1,e2,e3]T[e_1, e_2, e_3]^T.

Specific Gain Selections for Eigenaxis Rotation:
To achieve the "optimal" eigenaxis rotation (shortest path), the paper specifies:


Ideal Case (Perfect Inertia Knowledge):
D=dJD = d J
K=kJK = k J
Where dd and kk are positive scalar constants.


Gain Tuning Parameters:
The scalars dd and kk are derived from desired second-order system characteristics (natural frequency ωn\omega_n and damping ratio ζ\zeta):
k=ωn2k = \omega_n^2
d=2ζωnd = 2 \zeta \omega_n
(Note: For large angles >90∘> 90^\circ, the paper suggests using a modified settling time relation, but the linear approximation holds for gain derivation).


Sign Logic for Shortest Path (Remark 5):
To ensure rotation via the shortest angular path, the sign of the quaternion feedback must be adjusted based on the scalar part of the error quaternion (q4eq_{4e}):
u=−ω×Jω−Dω−sign(q4e)Kqveu = -\omega^\times J \omega - D \omega - \text{sign}(q_{4e}) K q_{ve}

If q4e<0q_{4e} < 0, flip the sign of the KK term to rotate the "other way" around the sphere.



5. Summary of Variables for Code Implementation
VariableTypeDescriptionSource EqJMatrix(3x3)Inertia Matrix(1)wVector(3)Angular Velocity(1)uVector(3)Control Torque Output(9)qVector(4)Current Attitude(3)qcVector(4)Commanded Attitude(6)qeVector(4)Error Quaternion(6)DMatrix(3x3)Damping Gain (dJdJ)(20)KMatrix(3x3)Stiffness Gain (kJkJ)(20)omega_nFloatNatural Frequency(35)zetaFloatDamping Ratio(35)
Pseudocode Logic Flow
def calculate_control_torque(J, w, q_current, q_target, d_scalar, k_scalar):
    # 1. Calculate Error Quaternion (assuming q_target is [0,0,0,1] for regulation)
    # If q_target is arbitrary, perform quaternion multiplication q_target_inv * q_current
    q_error = calculate_quaternion_error(q_current, q_target)
    
    q_error_vec = q_error[0:3]
    q_error_scalar = q_error[3]
    
    # 2. Determine Sign for Shortest Path
    sign_factor = 1.0 if q_error_scalar >= 0 else -1.0
    
    # 3. Construct Gain Matrices (Eigenaxis requirement: proportional to Inertia)
    D = d_scalar * J
    K = k_scalar * J
    
    # 4. Calculate Gyroscopic Term (Optional: set to 0 if mu=0)
    # Skew symmetric matrix of w
    w_skew = create_skew_symmetric(w)
    gyro_term = w_skew @ (J @ w)
    
    # 5. Apply Control Law (Eq 9)
    # u = - (gyroscopic_coupling) - D*w - sign*K*q_error_vec
    u = -gyro_term - (D @ w) - (sign_factor * (K @ q_error_vec))
    
    return u
This structure isolates the mathematical definitions from the logic, allowing an LLM to generate the specific matrix operations and integration loops required for your simulation.