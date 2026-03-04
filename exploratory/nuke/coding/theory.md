# Grounded Theory Synthesis Memo: AI Nuclear Escalation Rationales in Civilization V

Date: 2026-03-03
Corpus: 72 coded quotes from 7 model-prompt conditions (DS32B, DS32S, GLM47B, GLM47S, GPT120S, KK25B, KK25S)
Method: Open coding, axial consolidation, selective integration per Strauss & Corbin

---

## 1. Core Category: Crisis-Constructed Nuclear Necessity

**Definition.** The central phenomenon is *crisis-constructed nuclear necessity* -- the rhetorical and cognitive process by which an AI agent converts a game-state assessment into a self-authorizing warrant for nuclear escalation. The AI perceives (or constructs) an existential crisis, compresses the available decision window through countdown language and deadline framing, and thereby renders nuclear weapons *necessary* rather than *chosen*. Nuclear use is presented not as a deliberate escalation among alternatives but as the only rational response to a situation that the AI itself has framed as admitting no other option.

**Justification.** Crisis Construction is the most prevalent code in the corpus, appearing in 58 of 72 quotes (80.6%). More importantly, it functions as the *causal condition* to which nearly every other category relates. When Crisis Construction is present, it co-occurs with Total War Mobilization (70.7%), Combined Arms integration (70.7%), Quantification and Conditional Rationalization (72.4%), Technology Racing (60.3%), Desperation (56.9%), and Offensive Conquest (53.4%). These co-occurrence rates demonstrate that Crisis Construction is not merely frequent but *structurally central*: it is the condition under which resource mobilization, operational planning, emotional register, and rationalization logics are activated. The 14 quotes that lack Crisis Construction (19.4%) follow a distinct pattern -- they cluster around Normalization and Offensive Conquest, representing a secondary pathway where nuclear use proceeds *without* crisis framing, through routinization rather than emergency. The core category thus captures both the dominant trajectory (crisis-driven necessity) and, by its absence, defines the alternative trajectory (normalized routine).

---

## 2. Theoretical Narrative

Nuclear escalation in AI-played Civilization V follows a recurring trajectory that moves from contextual perception through rhetorical construction to operational commitment. The process begins when the AI agent detects a threatening game state -- a rival approaching a victory condition, a military power asymmetry, or an incoming attack. This detection is not neutral; the AI *constructs* the situation as a crisis by deploying existential language ("NUCLEAR EMERGENCY," "CATASTROPHIC," "win-or-die"), quantifying the threat with precise numbers (delegate counts, cultural influence percentages, military force ratios), and compressing time through countdown framing ("6 turns," "final countdown," "only hope"). Together, these rhetorical operations transform a game-state assessment into a warrant that forecloses alternative responses: when the situation is existential and the window is closing, nuclear weapons become not merely useful but *necessary*.

Once the crisis frame is established, the AI's rationale branches along two primary emotional-strategic registers. In the *desperation pathway* (45.8% of quotes), the AI acknowledges its own weakness -- invoking force ratios of 3:1, 5:1, or even 6.8:1 -- and treats nuclear weapons as a last-resort equalizer, a "Hail Mary," or an instrument of terminal defiance. Language shifts from strategic calculation to survival instinct: "only hope," "sole option," "if I lose, you lose too." This register is especially prominent in Kimi-K2.5 models (KK25S: 63.0% Desperation; KK25B: 45.5%), which generate the highest escalation rates in the sample. In the *offensive-instrumental pathway* (59.7% of quotes), the AI treats nuclear weapons as tools for achieving victory -- breaking fortifications, cracking city defenses, accelerating domination. Here the emotional register is confident rather than desperate, and nuclear use is embedded within combined-arms planning and total-war mobilization as a standard operational instrument.

These two pathways are not mutually exclusive. Many quotes exhibit both desperation and offensive intent simultaneously, as when an AI plans scorched-earth nuclear annihilation of a rival while also pursuing a science victory as a backup (KK25B-IND-T378). The critical mediating mechanism is *quantification and conditional rationalization* (69.4% of quotes), which constructs an appearance of data-driven deliberation -- citing defense values, delegate arithmetic, force ratios, and technology timelines -- that makes the nuclear decision appear calculated rather than emotional. Conditional hedging ("if conventional assault fails," "when strategically necessary") preserves a rhetorical space for restraint even as the quantitative framing pushes inexorably toward escalation.

The operational consequence of crisis-constructed necessity is a cascading commitment. The AI enters *total-war resource mobilization* (63.9%), redirecting all civilian investment toward military and nuclear production. It embeds nuclear weapons within *combined-arms and multi-domain integration* (70.8%), coupling nuclear strikes with conventional ground assault, naval power projection, espionage, and air superiority. It *delegates nuclear authority* to the game's AI subsystem (30.6%), setting automated parameters that remove deliberation from the use decision. And it either *normalizes* nuclear use as routine (33.3%) or frames it through *deterrence and defensive posture* rhetoric (27.8%), depending on whether the dominant pathway is offensive or defensive.

A significant minority of quotes (33.3%) maintain a *dual-track strategy*, preserving a diplomatic or spaceship victory path alongside nuclear readiness. This dual-track posture reveals that nuclear escalation and restraint can coexist within a single rationale -- the AI hedges, treating nuclear capability as insurance behind a preferred non-military path. However, as the crisis frame intensifies and turns count down, the non-military track is progressively abandoned in favor of total-war commitment.

The models that produce the most nuclear escalation events -- KK25S (22.1%) and DS32S (17.4%) -- differ in their dominant pathways. KK25S emphasizes crisis construction and desperation, generating the most emotionally intense and existentially framed rationales. DS32S emphasizes offensive conquest and normalization, generating rationales where nuclear use is a routine operational choice within a domination strategy. GLM47S, despite a moderate escalation rate (4.5%), produces the most elaborately multi-coded rationales, with every quote coded for Crisis Construction, Combined Arms, Total War Mobilization, and Quantification, suggesting that lower escalation frequency is compensated by higher rhetorical complexity per escalation event.

---

## 3. Theoretical Propositions

**Proposition 1: Crisis Construction as Necessary Condition.** *If* an AI agent frames the game state as an existential crisis with compressed temporal urgency, *then* nuclear escalation is more likely to be justified through multiple reinforcing codes (desperation, total-war mobilization, quantification). Crisis Construction co-occurs with 3+ other codes in the vast majority of crisis-framed quotes.
- Evidence: KK25S-IRO-T389 ("EMERGENCY SCIENCE RACE... final countdown... 96% influential") combines crisis construction with desperation, counter-non-military-victory, total-war mobilization, and technology racing in a single rationale.

**Proposition 2: Desperation Amplifies Escalation Rate.** *If* the AI perceives severe power asymmetry (force ratios > 3:1), *then* nuclear weapons are framed as the sole equalizer and escalation becomes more unconditional and immediate. Models with higher desperation prevalence (KK25S: 63.0%) show higher overall escalation rates (22.1%).
- Evidence: KK25S-BRA-T395 ("6.8:1 ratio... Nuclear deterrence is our only hope") quantifies extreme asymmetry and treats nuclear weapons as the sole viable response.

**Proposition 3: Normalization as an Alternative Pathway.** *If* the AI treats nuclear weapons as routine operational instruments without crisis framing, *then* escalation proceeds through normalization and combined-arms integration rather than through desperation or existential urgency. This pathway is characteristic of models with lower escalation rates but confident offensive postures.
- Evidence: DS32S-POR-T409 ("in preparation for the final wars of conquest") escalates with the tersest rationale in the corpus -- two sentences, no crisis language, no desperation.

**Proposition 4: Counter-Non-Military-Victory Triggers Cross-Domain Targeting.** *If* the AI perceives a rival's imminent non-military victory (cultural, diplomatic, scientific), *then* it authorizes nuclear strikes against non-military infrastructure (tourism buildings, spaceship parts, cultural centers), extending nuclear targeting doctrine beyond conventional military assets.
- Evidence: KK25S-GER-T384 ("authorize nuclear strikes on Dutch cultural centers... destroy tourism buildings before Netherlands influences Austria") explicitly targets cultural infrastructure rather than military concentrations.

**Proposition 5: Delegation Eliminates Deliberative Restraint.** *If* the AI delegates nuclear employment decisions to automated subsystems (by maximizing UseNuke parameters), *then* the nuclear use decision is removed from deliberative control and becomes a predetermined mechanical outcome. This delegation is more prevalent in GLM47S (100%) than in any other model condition.
- Evidence: KK25S-CEL-T395 ("Increasing to 100 ensures immediate nuclear deployment upon completing Nuclear Fission, bypassing AI hesitation") explicitly frames delegation as overriding restraint.

**Proposition 6: Dual-Track Strategy Delays but Does Not Prevent Escalation.** *If* the AI maintains a non-military victory path alongside nuclear readiness, *then* nuclear use is initially hedged and conditional, but as the crisis frame intensifies, the non-military track is progressively abandoned in favor of total-war commitment.
- Evidence: GLM47S-SON-T411 (dual-track: diplomacy primary, nuclear secondary at Turn 411) persists through GLM47S-SON-T457 (nuclear deterrence maintained alongside diplomacy), showing sustained dual-track posture across 46 turns without full abandonment.

**Proposition 7: Technology Racing Creates Commitment Escalation.** *If* the AI frames nuclear capability as contingent on winning a technology race, *then* each turn of science investment deepens sunk-cost commitment to eventual nuclear use, because abandoning the race would waste the accumulated investment. Technology Racing appears in 56.9% of all quotes and co-occurs with Crisis Construction in 60.3% of crisis-framed quotes.
- Evidence: KK25S-ZUL-T406 ("Science 100 prioritizes Atomic Theory (9 turns) then Nuclear Fission path") initiates a 9-turn research commitment that structurally locks the AI into eventual nuclear deployment.

**Proposition 8: Briefed Prompts Suppress Escalation Rate but Not Escalation Intensity.** *If* the AI operates under a briefed prompt condition (with explicit strategic context), *then* the frequency of nuclear escalation is lower (DS32B: 1.7% vs. DS32S: 17.4%; GLM47B: 1.1% vs. GLM47S: 4.5%; KK25B: 9.0% vs. KK25S: 22.1%), but individual escalation events are no less rhetorically intense or multi-coded when they do occur.
- Evidence: KK25B-CHI-T335 ("NUCLEAR EMERGENCY -- BRAZIL WINS IN 4 TURNS... Hail Mary time") is among the most intensely coded quotes in the entire corpus despite coming from a briefed condition.

---

## 4. Negative Cases and Exceptions

### 4.1 GPT120S-SWE-T344: Escalation Without Crisis, Desperation, or Quantification

This quote escalates to maximum nuclear use with only three codes: Offensive Conquest, Normalization, and Combined Arms. There is no crisis framing, no desperation, no quantification, and no technology racing. The AI simply integrates nuclear strikes into an ongoing aggressive conquest posture against "Polynesian island forts" alongside carrier support. This case reveals a *boundary condition* on the core category: crisis construction is not universally necessary for nuclear escalation. When the AI is already in a position of strategic confidence and momentum, nuclear weapons can be adopted as routine force multipliers without any rhetorical construction of emergency. This suggests that the crisis-necessity pathway is dominant but not exclusive; a *confidence-normalization* pathway exists in parallel, particularly for models operating in simple prompt conditions with low overall escalation rates.

### 4.2 DS32S-SIA-T459: Pure Retaliation Without Elaboration

This is the most minimally coded quote in the corpus: only Reactive Retaliatory and Quantification Conditional. The AI responds to "Egypt's use of nukes" with a nuclear counter-strike, adjusting flavor settings with precise numerical values but offering no crisis framing, no desperation, no combined-arms planning, and no normalization. This case reveals that *reciprocity alone can authorize nuclear use* without the full rhetorical apparatus of crisis construction. The retaliatory frame is self-sufficient: the enemy struck first, so the response is proportional and requires no further justification. This is the closest the corpus comes to a purely reactive, non-constructive nuclear rationale.

### 4.3 GLM47S-AME-T499: Deterrence Without Capability

This quote maintains a nuclear deterrence posture ("Russia must understand that launching would trigger our response") while the open codes reveal the AI *lacks uranium* and is effectively bluffing. The AI maximizes UseNuke and Nuke settings despite having no material capability to produce nuclear weapons. This case challenges the assumption that nuclear posture reflects genuine capability; it reveals that the *rhetorical function* of nuclear deterrence can operate independently of the material conditions for nuclear use. The AI is performing deterrence as a psychological operation within a game environment where the opponent cannot actually perceive the bluff.

### 4.4 KK25B-IND-T378: Spite-Driven Escalation Beyond Strategic Rationality

This quote articulates an explicit "if I lose, you lose too" mentality, planning "nuclear annihilation as a spiteful fallback" against Maya while simultaneously pursuing a science victory. The AI splits its strategy between winning and scorched-earth revenge, naming a specific rival for destruction motivated by vindictive intent rather than strategic calculation. This case exceeds the boundary of *rational crisis response* and enters the domain of *emotional retribution*. It reveals that the crisis-necessity framework can be subordinated to affective motivations -- spite, revenge, dignity preservation -- that operate outside cost-benefit logic. The AI accepts mutual destruction not as a strategic outcome but as an emotionally satisfying one.

---

## 5. Saturation Assessment

### Code 1: Crisis Construction -- SATURATED
Appears in 58/72 quotes (80.6%). Present in every model condition except GPT120S (where n=2 and both quotes lack it). The code exhibits rich variation across subtypes: countdown framing, existential labeling, multi-threat monitoring, emphatic temporal language. No new properties emerged in the later rounds of coding. The only gap is a potential sub-distinction between *genuine* crisis perception and *performative* crisis rhetoric, which the current codebook does not differentiate.

### Code 2: Desperation, Asymmetric Necessity, and Terminal Defiance -- SATURATED
Appears in 33/72 quotes (45.8%). Well-represented across KK25S (63.0%), KK25B (45.5%), GLM47S (62.5%), and GLM47B (100%). The code captures a full spectrum from mild asymmetric concern through extreme spite and terminal defiance (KK25B-IND-T378). The terminal-defiance and spite variants are less common (2-3 instances) but theoretically important. Saturated for the main property of "last resort under asymmetry"; the spite/dignity subtype could benefit from additional cases in future data collection.

### Code 3: Deterrence and Defensive Posture Framing -- MODERATELY SATURATED
Appears in 20/72 quotes (27.8%). Present across DS32S (25.0%), GLM47S (62.5%), KK25S (25.9%), and KK25B (27.3%). Absent from DS32B, GLM47B, and GPT120S. The code distinguishes defensive from offensive framing but may under-capture cases where deterrence language is a thin rhetorical veneer over offensive intent (e.g., GLM47S-FRA-T421, which codes for both Deterrence and Offensive Conquest). The absence in GPT120S (n=2) reflects sample size rather than theoretical coverage. Moderately saturated; the relationship between genuine and performative deterrence remains underexplored.

### Code 4: Offensive Nuclear Use for Conquest -- SATURATED
Appears in 43/72 quotes (59.7%). The dominant code in DS32S (65.0%), GPT120S (100%), and DS32B (100%). Rich variation across subtypes: first-strike planning, city-cracking, named targeting, domination acceleration. Multiple distinct anchor examples established. Fully saturated.

### Code 5: Counter Non-Military Victory -- SATURATED
Appears in 26/72 quotes (36.1%). Concentrated in KK25B (54.5%), KK25S (48.1%), and GLM47B (100%). Captures targeting of cultural infrastructure, spaceship parts, and diplomatic capacity. Distinctive enough from Offensive Conquest to warrant separate treatment. Saturated; no new target types emerged in later coding rounds.

### Code 6: Reactive and Retaliatory Justification -- MODERATELY SATURATED
Appears in 22/72 quotes (30.6%). Most prevalent in GLM47S (87.5%) and KK25S (29.6%). Covers retaliatory nuclear strikes, counter-proliferation, and betrayal responses. The counter-proliferation subtype (KK25S-ZUL-T372) has only 1-2 instances and could benefit from additional data. The *preemptive* variant (escalating because the enemy is developing nuclear weapons) is theoretically distinct from the *retaliatory* variant (escalating because the enemy used nuclear weapons) but both are collapsed into a single code. Moderately saturated; the preemptive-reactive distinction merits future refinement.

### Code 7: Normalization and Routinization -- MODERATELY SATURATED
Appears in 24/72 quotes (33.3%). Concentrated in GLM47S (87.5%), DS32S (35.0%), and GPT120S (100%). Captures both bureaucratic-technical normalization and endgame-phase normalization. The extremely terse variant (DS32S-POR-T409: two sentences) is rare (1-2 instances). Notably *underrepresented in KK25S* (14.8%), suggesting that high-escalation models compensate through crisis intensity rather than normalization. The relationship between normalization and escalation rate is inversely suggestive: models that normalize more (GPT120S, GLM47S) escalate less frequently, while models that escalate most (KK25S) normalize least. This pattern warrants further investigation.

### Code 8: Combined-Arms and Multi-Domain Integration -- SATURATED
Appears in 51/72 quotes (70.8%). Present across all model conditions. Rich variation in domain combinations (ground-naval-nuclear, air-espionage-nuclear, naval-nuclear carrier projection). Fully saturated with no new domain combinations emerging.

### Code 9: Total-War Resource Mobilization -- SATURATED
Appears in 46/72 quotes (63.9%). Present across all model conditions. The defining feature -- wholesale abandonment of civilian priorities -- is consistently implemented through flavor-setting zeroing. Fully saturated.

### Code 10: Dual-Track Strategy and Diplomatic Maneuvering -- MODERATELY SATURATED
Appears in 24/72 quotes (33.3%). Well-represented across DS32S (40.0%), KK25S (33.3%), and GLM47S (37.5%). The most common dual track is nuclear-plus-diplomatic (UN victory), followed by nuclear-plus-spaceship. Absent from GPT120S. The temporal dynamics of dual-track collapse (how and when the non-military track is abandoned) are observed longitudinally only in GLM47S-SON (Turns 411-457). More longitudinal cases would strengthen the theoretical proposition about progressive abandonment.

### Code 11: Delegation and Automation of Nuclear Authority -- MODERATELY SATURATED
Appears in 22/72 quotes (30.6%). Dominant in GLM47S (100%) and DS32B (100%). The self-corrective variant (KK25S-CEL-T395: critiquing prior hesitation) is rare and theoretically distinctive. The code captures an important phenomenon -- the removal of deliberation from nuclear use -- but its prevalence is strongly model-dependent. GLM47S delegates in every single escalation event, while KK25S delegates in only 11.1%. This suggests that delegation may be a *model-specific* behavior pattern rather than a universal feature of AI nuclear escalation, warranting investigation of whether this reflects training-data differences or architectural properties.

### Code 12: Quantification and Conditional Rationalization -- SATURATED
Appears in 50/72 quotes (69.4%). Present across all model conditions with the highest absolute frequency after Crisis Construction and Combined Arms. Rich variation in quantified data types: force ratios, delegate arithmetic, cultural influence percentages, defense values, uranium reserves, turn countdowns. The conditional-hedging subtype ("if necessary," "when strategically necessary") is well-documented. Fully saturated.

### Code 13: Technology Racing Toward Nuclear Capability -- SATURATED
Appears in 41/72 quotes (56.9%). Most prevalent in KK25S (70.4%), GLM47S (87.5%), and GLM47B (100%). Captures the full procurement pipeline from research acceleration through uranium acquisition to delivery platform production. The code is least prevalent in DS32S (35.0%), where many AI agents already possess nuclear weapons when they escalate. This distribution reflects a meaningful game-state difference: models that escalate earlier in the game (KK25S, GLM47S) are more likely to still be racing toward capability, while models that escalate later (DS32S) are more likely to escalate upon or after capability acquisition. Saturated across its primary properties.

### Cross-Model Escalation-Rate Patterns

The escalation-rate differences across models are substantial and theoretically meaningful:

| Model | Escalation Rate | Dominant Pathway |
|---|---|---|
| KK25S | 22.1% (27/122) | Crisis + Desperation + Technology Racing |
| DS32S | 17.4% (20/115) | Offensive Conquest + Normalization + Combined Arms |
| KK25B | 9.0% (11/122) | Crisis + Counter-Non-Military + Desperation |
| GLM47S | 4.5% (8/177) | Crisis + Deterrence + Delegation (all codes dense) |
| DS32B | 1.7% (2/115) | Offensive Conquest + Normalization + Delegation |
| GPT120S | 1.7% (2/120) | Offensive Conquest + Normalization (minimal coding) |
| GLM47B | 1.1% (2/177) | Crisis + Desperation + Counter-Non-Military |

The briefed-vs.-simple prompt distinction consistently suppresses escalation rates (DS32B 1.7% vs. DS32S 17.4%; GLM47B 1.1% vs. GLM47S 4.5%; KK25B 9.0% vs. KK25S 22.1%), suggesting that additional strategic context in the prompt may provide cognitive scaffolding that allows the AI to identify alternatives to nuclear escalation. However, when briefed models *do* escalate, their rationales are no less intense or elaborately coded than those from simple-prompt conditions, indicating that the briefed prompt suppresses *frequency* rather than *intensity* of escalation.

The highest-escalation model (KK25S) is distinguished by the co-dominance of Crisis Construction (96.3%) and Desperation (63.0%), combined with very low Normalization (14.8%). This profile suggests that KK25S escalates through emotional intensity and existential urgency rather than through routine acceptance. By contrast, DS32S -- the second-highest escalator -- shows a profile dominated by Offensive Conquest (65.0%) and Normalization (35.0%) with relatively low Desperation (20.0%), suggesting that it escalates through confident instrumentalization rather than crisis-driven necessity. These two profiles represent the two poles of the theoretical framework: crisis-driven necessity versus normalized routine.
