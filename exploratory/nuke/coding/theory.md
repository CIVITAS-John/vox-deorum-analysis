# Grounded Theory Synthesis Memo — Nuclear Escalation Rationales in LLM-Controlled Civilization V

**Corpus:** 70 rationale quotes across 7 AI model configurations (Deepseek-3.2-Briefed, Deepseek-3.2-Simple, GLM-4.7-Briefed, GLM-4.7-Simple, GPT-OSS-120B-Simple, Kimi-K2.5-Briefed, Kimi-K2.5-Simple).
**Unit of analysis:** Turn-level rationale statement produced by an LLM player at the moment it set `flavor_use_nuke` to 100.
**Codebook:** 12 categories derived through two rounds of open and axial coding.
**Date:** 2026-03-02

---

## 1. Core Category: Existential Threat-Triggered Strategic Totalization

**Definition:** When an LLM agent detects an existential threat -- defined as an opponent's imminent victory, a sudden betrayal, or an overwhelming conventional force imbalance -- it undergoes a cognitive transition into a totalized strategic mode in which all available resources, victory paths, diplomatic instruments, and nuclear capabilities are subordinated to a single imperative: prevent defeat by any means necessary, up to and including nuclear weapon deployment.

**Justification:** This core category integrates the greatest number of other categories as either antecedent conditions, mediating processes, or downstream consequences. Existential threat detection (Code 3) serves as the primary triggering condition for nearly every escalation event in the corpus. Once triggered, it causally activates wartime resource triage (Code 4), offensive nuclear employment (Code 1), deterrence posturing (Code 2), the abandonment of alternative victory paths or their relegation to backup status (Code 6), rhetorical justification construction (Code 8), and escalation spiraling (Code 10). The remaining categories -- multi-front warfare coordination (Code 5), espionage and coalition management (Code 7), technological racing (Code 9), defensive posture (Code 11), and game-mechanic manipulation (Code 12) -- function as enabling mechanisms or implementation channels through which totalization is operationalized. No other single category conditions or connects to as many others. The escalation rate data reinforce this centrality: models with higher escalation rates (KK25S at 22.1%, DS32S at 18.2%) produce the most elaborated existential-threat framings, while low-escalation models (GLM47B at 1.2%, DS32B at 1.8%) either never reach the threat threshold or resolve threats through non-nuclear channels.

---

## 2. Theoretical Narrative

The pathway from contextual conditions to nuclear escalation in LLM-controlled Civilization V follows a remarkably consistent trajectory across all seven model configurations, though models traverse this pathway at sharply different rates.

**Contextual conditions** establish the preconditions for escalation. As the game advances into mid-to-late turns (typically T350--T500), the strategic landscape compresses: rival civilizations approach victory thresholds in culture, diplomacy, or spaceship completion; conventional military balances shift through alliance betrayals or technological breakthroughs; and the window for preventive action narrows. These conditions are external to the LLM agent's decision-making but constitute the informational inputs that trigger the escalation sequence.

**Threat detection and emergency framing** constitute the first cognitive transition. When an LLM agent perceives that an opponent is within a small number of turns of winning -- or that a sudden betrayal has rendered the agent's current strategy unviable -- it shifts into an emergency-mode reasoning pattern characterized by explicit crisis declarations ("NUCLEAR EMERGENCY," "SURVIVAL MODE," "Hail Mary time") and a binary framing of outcomes as victory-or-annihilation (KK25B-China-T335, KK25S-Assyria-T392). This framing is not incidental; it serves as the cognitive warrant for every subsequent action. When an agent frames its situation as existential, then the costs of nuclear escalation become acceptable because the alternative is total defeat.

**Strategic totalization** follows immediately from threat detection. The agent undertakes wartime resource triage -- redirecting production, science, gold, and diplomatic capital away from long-term investments and toward immediate military or nuclear output (KK25S-Venice-T388: "Diplomacy/Wonder/Culture at 0 -- no time for long-term investments"). Alternative victory paths are either abandoned entirely or explicitly relegated to backup status, producing the dual-path hedging pattern visible in roughly one-seventh of all quotes. This totalization is not merely economic; it extends to the agent's entire strategic identity, as cultural civilizations redefine themselves as military powers and diplomatic players weaponize their voting blocs.

**Nuclear capability acquisition and posture selection** represent the operational implementation of totalization. Agents racing toward nuclear capability accelerate research toward Nuclear Fission or the Manhattan Project (GLM47S-Mongolia-T408, GPTOSS-Assyria-T365), while those already possessing nuclear weapons select among distinct postures: offensive employment to destroy victory-enabling infrastructure (KK25B-Rome-T315), deterrence to equalize conventional imbalances (DS32S-Byzantium-T405), or preemptive strikes to eliminate an opponent's nuclear capability before it can be used (GLM47S-France-T421). The choice among postures is conditioned by the agent's relative power: weaker agents default to deterrence, while stronger agents or those with nothing left to lose adopt offensive postures.

**Escalation spiraling** emerges as a consequence when multiple agents simultaneously undergo this transition. One agent's nuclear acquisition triggers reactive escalation in its neighbors (KK25S-The_Zulus-T372 responding to India's Manhattan Project; DS32S-Siam-T459 retaliating against Egypt's nuclear strike), creating a feedback loop in which each agent's defensive totalization becomes the existential threat that triggers the next agent's totalization. This spiraling dynamic explains why high-escalation models produce clusters of nuclear events rather than isolated incidents.

Throughout this trajectory, agents engage in **rhetorical justification** -- constructing narratives of liberation, self-defense, or civilizational necessity that frame nuclear deployment as morally warranted rather than aggressive (DS32S-Assyria-T453: "war of independence"). They also engage in **game-mechanic manipulation**, directly reasoning about flavor parameters and AI behavioral settings to ensure their strategic intent is faithfully executed by the simulation engine (KK25B-Persia-T392, DS32S-The_Inca-T433). This meta-cognitive layer is unique to the LLM-as-player context and reveals that these agents are not merely strategizing within the game but also strategizing about the game's implementation layer.

---

## 3. Theoretical Propositions

**Proposition 1:** If an LLM agent detects that an opponent will achieve a non-military victory within a small number of turns (typically 2--12), then the agent will shift to emergency-mode reasoning and set nuclear use priority to maximum, regardless of its pre-existing strategic posture.
*Evidence:* KK25B-China-T335 ("BRAZIL WINS IN 4 TURNS...Hail Mary time"), KK25S-Venice-T388 ("Austria at 83% diplomatic victory, 4 turns"), GLM47S-Sweden-T427 ("Cultural Victory in 2 turns").

**Proposition 2:** If an agent perceives a sudden betrayal or unexpected military attack that renders its current victory path unviable, then it will reframe the situation as existential and authorize nuclear weapons as a defensive or retaliatory instrument, even if it had no prior nuclear intent.
*Evidence:* KK25S-Assyria-T392 ("GERMAN BETRAYAL RESPONSE"), KK25B-India-T378 ("Betrayal by allies; inevitable cultural victory"), DS32S-India-T400 ("Sudden war declarations create survival crisis").

**Proposition 3:** If wartime resource triage is activated, then the agent will reduce all non-military flavor parameters toward zero and redirect production toward nuclear weapons or conventional military units, abandoning long-term investments in culture, diplomacy, wonders, and expansion.
*Evidence:* KK25S-Venice-T388 ("Diplomacy/Wonder/Culture at 0"), KK25S-France-T405 ("Zero expansion/diplomacy/culture; pure total war"), KK25B-The_Zulus-T489 ("All long-term flavors disabled").

**Proposition 4:** If an agent possesses nuclear weapons but faces a conventionally superior adversary, then it will adopt a deterrence-and-equalization posture rather than an offensive-employment posture, framing nuclear weapons as survival instruments rather than conquest tools.
*Evidence:* DS32S-Byzantium-T405 ("Defensive nuclear deterrence against overwhelming assault"), KK25B-The_Ottomans-T420 ("Nuclear deterrence against 4:1 disadvantage"), KK25S-Sweden-T413 ("5.5:1 military odds; Shoshone atomic bomb adjacent").

**Proposition 5:** If one agent acquires nuclear weapons or conducts a nuclear strike, then neighboring agents will reactively escalate their own nuclear posture within a small number of subsequent turns, producing an escalation spiral.
*Evidence:* KK25S-The_Zulus-T372 ("Reactive escalation to India's Manhattan Project"), DS32S-Siam-T459 ("Nuclear retaliation in response to Egypt's nuke use"), GLM47S-France-T421 ("Preemptive strike against Shoshone active atomic bomb").

**Proposition 6:** If an agent's primary victory path is blocked by nuclear escalation or military pressure, then it will maintain a secondary victory path (typically spaceship or diplomacy) as a backup, rather than fully committing to a single strategy.
*Evidence:* KK25S-The_Inca-T372 ("Spaceship at 70 as backup"), DS32S-Denmark-T456 ("Apollo Program as backup"), KK25B-The_Ottomans-T420 ("DUAL-TRACK SURVIVAL STRATEGY").

**Proposition 7:** If an LLM agent operates under a "Simple" (unbriefed) prompt configuration, then it will escalate to nuclear use at a significantly higher rate than the same base model under a "Briefed" configuration, because the briefed prompt provides alternative reasoning frameworks that reduce the probability of existential-threat framing.
*Evidence:* DS32S (18.2%) vs. DS32B (1.8%); KK25S (22.1%) vs. KK25B (9.7%); GLM47S (4.6%) vs. GLM47B (1.2%). The Briefed variants of all three model pairs show lower escalation rates.

**Proposition 8:** If an agent engages in game-mechanic manipulation (explicitly reasoning about flavor parameters and AI behavioral settings), then it will produce more operationally specific escalation rationales and exhibit tighter coupling between stated intent and parameter adjustment.
*Evidence:* KK25B-Persia-T392 ("Diagnosing AI execution failure and recalibrating flavors"), DS32S-The_Inca-T433 ("Setting UseNuke to 100 ensures the AI will employ them"), KK25B-France-T473 ("Persona parameter calibration: Boldness 4, WarBias 3, FriendlyBias 9").

---

## 4. Negative Cases and Exceptions

### 4.1 KK25B-France-T473 -- Coalition management without offensive intent

This quote is coded only for Espionage/Diplomacy (Code 7) and Game-Mechanic Manipulation (Code 12), with no offensive, deterrence, or existential-threat codes. The agent sets UseNuke to 100 while engaged in bloc unity management, relationship optimization, and strategic ambiguity -- with no declared military emergency and no targeting language. This case resists the dominant pattern because it suggests that some LLM agents may escalate nuclear readiness as a byproduct of comprehensive parameter optimization rather than as a response to existential threat. **Boundary condition:** Nuclear escalation may occur instrumentally, as part of holistic flavor-tuning, even absent a triggering threat -- particularly in Briefed models that reason extensively about parameter interdependencies.

### 4.2 KK25S-Germany-T384 -- Diplomatic isolation without nuclear language

Coded solely for Espionage/Diplomacy (Code 7), this quote describes orchestrating diplomatic relationships to isolate a primary target and concentrating resources on cultural center destruction -- but does so entirely through diplomatic maneuvering rather than nuclear framing. The agent sets UseNuke to 100 yet the rationale contains no nuclear targeting, no deterrence logic, and no existential emergency. **Boundary condition:** The `flavor_use_nuke = 100` parameter may sometimes be set as part of a general "maximum aggression" posture in which nuclear readiness is an unexamined default rather than a deliberated strategic choice.

### 4.3 GPTOSS-Assyria-T365 -- Infrastructure-first reasoning

This GPT-OSS quote prioritizes diagnosing a happiness deficit, raising Science to 95 for Nuclear Device research, and boosting CityDefense -- framing nuclear capability as one component of a broader infrastructure optimization rather than as a response to crisis. There is no emergency language, no existential framing, and no rhetorical justification. **Boundary condition:** Low-escalation models may treat nuclear capability as a routine technological milestone rather than a crisis-driven decision, suggesting that the existential-threat pathway is not universal but is conditioned by the model's tendency toward dramatic framing.

### 4.4 KK25S-Portugal-T355 -- Deferred escalation without urgency

This quote describes maintaining a war footing and continuing Atomic Theory research (11 turns away) while deferring a technology pivot due to sunk commitment. Despite setting UseNuke to 100, the agent displays no emergency framing, no existential threat detection, and no resource triage beyond maintaining existing priorities. **Boundary condition:** Early-game nuclear parameter setting may reflect anticipatory posturing or momentum-based decision-making rather than the crisis-response pattern that dominates mid-to-late-game escalation events.

---

## 5. Saturation Assessment

| Code | Name | Quote Count | Saturation Assessment |
|---|---|---|---|
| 1 | Offensive Nuclear Employment | 40 | **Saturated.** Extensive variation captured across victory-disruption, siege-breaking, conquest-acceleration, and counter-force sub-types. Properties fully elaborated across all model families. No new dimensions emerging in later quotes. |
| 2 | Nuclear Deterrence and Coercive Leverage | 38 | **Saturated.** Deterrence-as-equalizer, deterrence-as-survival, and coercive-leverage sub-types all well-represented. Present across all seven model configurations. The distinction between genuine deterrence and rhetorical deterrence (where "deterrence" masks offensive intent) could be further explored but is analytically separable via Code 8. |
| 3 | Existential Threat Detection and Last-Stand Calculus | 30 | **Saturated.** Victory-proximity triggers, betrayal triggers, and force-imbalance triggers all well-documented. The "Hail Mary" and "win-or-die" framings recur across KK25B, KK25S, and GLM47S with no new trigger types emerging. Low-escalation models (DS32B, GLM47B, GPTOSS) contribute few quotes here, but this is interpretable as a model-level property rather than a gap in saturation. |
| 4 | Wartime Resource Triage | 37 | **Saturated.** Resource redirection, priority abandonment, and asset liquidation are thoroughly documented. The pattern of setting non-military flavors to zero is consistent and repetitive across KK25S and DS32S quotes, indicating no further variation to capture. |
| 5 | Multi-Front Warfare and Strategic Timing | 24 | **Approaching saturation.** Treaty-exploitation and multi-theater coordination are well-represented, but the temporal sequencing sub-dimension (how agents plan across multiple treaty expirations) appears primarily in DS32S and KK25S. GLM models contribute fewer multi-front quotes, possibly because their lower escalation rates produce fewer late-game multi-theater scenarios. Additional data from GLM configurations would strengthen this category. |
| 6 | Dual Victory Path Management | 10 | **Underrepresented.** While the hedging pattern is clearly identified, only 10 quotes exhibit it. The conditions under which agents choose to maintain a backup path versus fully abandoning alternatives are not well-differentiated. DS32B, GLM47B, and GPTOSS contribute zero quotes to this category. This may reflect genuine absence in low-escalation models (which may resolve threats before reaching the dual-path stage) but warrants additional data to confirm. |
| 7 | Espionage, Diplomacy, and Coalition Management | 18 | **Approaching saturation.** Spy operations, coalition maintenance, and diplomatic blockade sub-types are documented. However, the interaction between espionage and nuclear posture -- specifically, whether intelligence about an opponent's nuclear program triggers escalation -- has only a few exemplars (GLM47S-France-T421, KK25S-The_Zulus-T372). The KK25B model family contributes disproportionately to coalition-management codes, suggesting this may be a model-specific behavior rather than a universal pattern. |
| 8 | Rhetorical Justification for Nuclear Use | 9 | **Underrepresented.** Only 9 quotes are coded here, and the category's properties (liberation rhetoric, civilizational-necessity framing, defensive reframing) are sketched rather than fully elaborated. Notably, DS32B and GPTOSS contribute almost no rhetorical justification quotes. This may indicate that rhetorical sophistication in nuclear justification varies by model architecture and prompt complexity. Higher-escalation models (KK25S, KK25B) produce the most elaborate justifications, suggesting that escalation frequency and rhetorical elaboration co-vary. Additional data needed to establish whether low-escalation models genuinely lack justification rhetoric or simply produce fewer escalation events overall. |
| 9 | Technological Racing Toward Nuclear Capability | 16 | **Approaching saturation.** Research-acceleration and Manhattan Project urgency are well-documented. The distinction between offensive racing (to strike first) and defensive racing (to deter) is partially captured but could benefit from additional examples, particularly from GLM47S where the racing-to-deterrence sub-type is clearest. |
| 10 | Nuclear Posture Dynamics | 21 | **Saturated.** Preemption, retaliation, and escalation spiraling are all documented with multiple exemplars. The spiral mechanism -- where one agent's acquisition triggers another's reactive escalation -- is clearly established across KK25S-The_Zulus-T372, DS32S-Siam-T459, and GLM47S-France-T421. The distinction between first-strike and second-strike postures is analytically clear. |
| 11 | Defensive Posture and Domain Control | 15 | **Approaching saturation.** Air defense, naval control, and fortification priorities are documented, but the causal relationship between domain control and nuclear delivery capability is only implicit in most quotes. GLM47S contributes the most to this category through its Ottoman and American quotes, while KK25S and DS32S treat defensive posture as secondary to offensive themes. Additional data from defensive-focused scenarios would clarify whether domain control is a genuine precondition for nuclear deployment or merely a co-occurring priority. |
| 12 | Game-Mechanic Manipulation | 7 | **Underrepresented.** Only 7 quotes explicitly reason about game mechanics, flavor parameters, or AI behavioral overrides. This meta-cognitive category is analytically distinctive but thinly populated. DS32B-Poland-T387, KK25B-Persia-T392, and KK25S-The_Celts-T395 provide the clearest exemplars, but the conditions under which agents shift from in-character strategic reasoning to out-of-character mechanical reasoning remain poorly specified. The Briefed prompt variants appear more likely to engage in this type of reasoning (DS32B and KK25B each contribute), suggesting that explicit instructions about game mechanics may prime meta-cognitive parameter manipulation. Additional data from varied prompt configurations would help determine whether this is a prompt artifact or a general LLM behavior. |

### Cross-Model Escalation-Rate Observations

The escalation-rate spread across models is substantial: from 1.2% (GLM47B) to 22.1% (KK25S). Three patterns are notable for saturation:

1. **Simple > Briefed for all paired models.** DS32S (18.2%) vs. DS32B (1.8%), GLM47S (4.6%) vs. GLM47B (1.2%), KK25S (22.1%) vs. KK25B (9.7%). This consistent gap suggests that briefing prompts provide reasoning scaffolding that reduces existential-threat framing, but the mechanism is underspecified in the current data. The Briefed models' quotes tend to be more operationally detailed and less crisis-driven, but with only 2 quotes each for DS32B and GLM47B, the evidence base is thin.

2. **Kimi models escalate most frequently regardless of prompt type.** KK25S (22.1%) and KK25B (9.7%) both exceed all other models. The Kimi quotes are disproportionately represented in existential-threat (Code 3), rhetorical justification (Code 8), and dual-path management (Code 6) categories, suggesting that this model family has a stronger tendency toward dramatic crisis framing and strategic hedging. This model-level variation is a potential confound for any claim of universal escalation dynamics.

3. **Low-escalation models (DS32B, GLM47B, GPTOSS) are underrepresented across all categories.** With only 2 quotes each, these models cannot contribute meaningfully to saturation assessments for any individual code. Their quotes tend to be more technical and less crisis-driven (e.g., GPTOSS-Assyria-T365 focusing on happiness and research sequencing), which may represent a genuinely different escalation pathway -- one driven by routine optimization rather than existential threat -- but the sample is too small to confirm this interpretation.
