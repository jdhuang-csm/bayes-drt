# Surface Description of Repository

This repository contains the generalized EIS inversion (gEISi) algorithm, as described in [1]. We recommend reading through the sections titled "Overview of the concept", "Design of algorithm", "Generalized EIS inversion problem", and "Generalized EIS inversion algorithm" before attempting to use this code.

TLDR: EIS data often exhibit non-ideal behavior, e.g., suppressed semicircles or suppressed ray in the Nyquist plot. This is conventionally understood as arising from a distribution(s) of parameter values, each giving rise to a different characteristic timescale. EIS inversion algorithms attempt to find this distribution(s) given a finite of data.

Normally, this would be impossible, because the distribution is a function (albeit one that is constrained to have certain properties, e.g., non-negative and vanishes for extreme parameter values), and the space of functions is infinite, but we only have a finite number of data points. Inversion attempts to do so by ascribing a prior probability for each such function, assigned based on some quality of that function. 

A popular example is the first-order Tikhonov regularization, in which the prior probability is set to be lower for functions with bigger first derivatives. Another famous one is the maximum entropy method, in which the prior probability is set to be lower for functions with lower entropy.

In [1], we review and discuss several assumptions underlying existing EIS inversion algorithms, and then proceed to make (among others) the two following generalizations:
(a) It is possible to write EIS inversion algorithms for models more complicated than the usual relaxation model (parallel (RC) circuit). See, for example, [2] and [3].
(b) In most contexts, the prior probability should be highest for Dirac delta distributions.

There are a fair amount of requirements for using the gEISi algorithm. The user needs to have a distributed model of the form of Equation (66), an example of which is shown in the section "Generalized EIS inversion problem" of reference [1]. The user should also reflect on whether the prior probability should be highest for the Dirac delta distribution for the user's application. If not, the gEISi algorithm is not suitable.

One last warning: the gEISi algorithm is ultimately a stochastic data-fitting tool with several underlying assumptions. We do not promise magic results - it does not make a bad model good. For a guided sample problem, see the word document attached, titled "Examples.docx".

Cheers,

Surya Effendy

References:

[1] Effendy, Surya, Juhyun Song, and Martin Z. Bazant. "Analysis, Design, and Generalization of Electrochemical Impedance Spectroscopy (EIS) Inversion Algorithms." Journal of the Electrochemical Society (2020).

[2] Song, Juhyun, and Martin Z. Bazant. "Electrochemical impedance imaging via the distribution of diffusion times." Physical review letters 120.11 (2018): 116001.

[3] Florsch, Nicolas, André Revil, and Christian Camerlynck. "Inversion of generalized relaxation time distributions with optimized damping parameter." Journal of Applied Geophysics 109 (2014): 119-132.
