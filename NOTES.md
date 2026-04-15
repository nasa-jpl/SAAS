Hello everyone,

 

As a reminder, the following was our proposed plan for this FY of the SURP. With an eventual goal of getting to integrating flight software (on hardware if possible), the main focus this year was to compare different FP approaches with higher fidelity models. Josef has done literature review and we have been having discussions to address the Q2 milestone, but we need to get started implementing the new FP methods. I’m happy to make a flyby mission what we model so we can incorporate Josef’s new rendering capability. I therefore propose the following as next steps:

    Josef creates the flyby scenario in SAAS. The flyby will assume the spacecraft had been sent on the desired trajectory and no TCMs are required after the start of the simulation. The spacecraft would be tasked with taking nadir images of the asteroid surface during the flyby, which necessitates high rate slews. A power-related fault is injected that simulates a temporary brownout that knocks out whichever reaction wheel is drawing the most current/applying the most torque and one star tracker is also knocked out. Onboard FP needs to respond to maximize the nadir imaging despite the loss of those components. Since it’s a power fault affecting GNC, there’s the possibility for cascading failures (leading to the second to last  bullet point).
    With that scenario, Josef implements a baseline “just safe” scenario that halts the imaging and points the antenna to Earth (JPL’s usual safe and phone home architecture).
    Josef also implements a “keep imaging, but swap to spare string hardware” strategy to keep going with the imaging, and once past the asteroid, points the antenna to Earth in a “need assistance” mode.
    Josef implements a new learning-based approach that identifies the issue as a power fault and not a fault with the wheel and star tracker, and as a result sheds loads by turning off non-essential hardware (eg. the spares) to prevent the fault from cascading.
    While all the above is happening, we keep discussing other FP architectures to implement, particularly those being researched by Steve and Ryan.

 

What are your thoughts for that plan?

 

Thanks,

David

 