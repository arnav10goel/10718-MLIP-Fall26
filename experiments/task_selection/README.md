# Task selection for recording

Which COIN tasks to record ourselves, and which COIN videos to watch first.

**How tasks were ranked.** We counted, for each COIN task, how many videos on the Hugging Face mirror share the exact same ordered steps. Then we kept tasks that are cheap, safe and doable at home, and where a mistake changes how the scene looks. The DTW baseline compares how frames look, so a mistake it cannot see, it cannot flag.

**How videos were ranked** (annotations only, no download): shorter task section, more of it covered by labelled steps (less dead time), step lengths more even, and `training` split first (the runner only accepts a training video as reference). Every link below was live on YouTube on 2026-10-05. Watch them and pick by eye: camera angle and clear hands matter more than this score.

Times are seconds into the YouTube video. `cov` is the share of the task section covered by labelled steps.

## MakeMatchaTea

64 matching videos. Steps: add tea powder → brew tea and stir.

Best overall. Cheap, safe, 64 matching videos. Skipping the powder leaves the cup clear, so the mistake is easy to see. Only 2 steps, so you cannot test steps done out of order.

| Video | Split | Task length (s) | cov | Step times (s) | Title |
| --- | --- | --- | --- | --- | --- |
| [JbHMReMOjDw](https://www.youtube.com/watch?v=JbHMReMOjDw&t=42s) | training | 28 | 0.86 | 42–56, 57–67 | How to Make Matcha Tea |
| [b-FCKeWlVLc](https://www.youtube.com/watch?v=b-FCKeWlVLc&t=41s) | training | 24 | 0.8 | 41–49, 51–62 | How to Make Matcha - Prepare Japanese Matcha Tea |
| [rNVoxiaITgI](https://www.youtube.com/watch?v=rNVoxiaITgI&t=3s) | training | 34 | 0.81 | 3–12, 15–34 | How to Make Ceremonial Grade Matcha Tea At Home |
| [13uVij4DsZk](https://www.youtube.com/watch?v=13uVij4DsZk&t=15s) | testing | 51 | 0.75 | 15–31, 39–61 | How to Make the Perfect Cup of Matcha Tea, Den's |
| [6tM6nrYaiHc](https://www.youtube.com/watch?v=6tM6nrYaiHc&t=23s) | testing | 23 | 0.61 | 23–29, 32–40 | Matcha Syrup Recipe |

## CookOmelet

20 matching videos. Steps: pour the egg into the pot → fry eggs → pour the egg onto the plate.

Strong pick. In COIN this task is really "fry an egg" (sunny-side up), not a folded omelet. 3 steps, so order mistakes are testable. Step 2 (fry) is much longer than the others.

| Video | Split | Task length (s) | cov | Step times (s) | Title |
| --- | --- | --- | --- | --- | --- |
| [382uPalxYhk](https://www.youtube.com/watch?v=382uPalxYhk&t=8s) | training | 58 | 0.93 | 8–13, 14–50, 52–65 | Frying Egg without oil in a Kyocera Ceramic Coated |
| [jGM8EImhpXg](https://www.youtube.com/watch?v=jGM8EImhpXg&t=36s) | training | 53 | 0.74 | 36–40, 41–70, 74–80 | How to fry an egg |
| [9Nr5JXHo9sQ](https://www.youtube.com/watch?v=9Nr5JXHo9sQ&t=48s) | training | 87 | 0.81 | 48–92, 93–100, 104–124 | How to "Fry" Eggs on the George Foreman Grill. |
| [4MhzKI4q5FE](https://www.youtube.com/watch?v=4MhzKI4q5FE&t=70s) | training | 42 | 0.68 | 70–71, 74–93, 94–103 | Perfect Sunny-Side Up Eggs |
| [GAN1s0AhZ-4](https://www.youtube.com/watch?v=GAN1s0AhZ-4&t=4s) | testing | 53 | 0.73 | 4–5, 6–40, 44–48 | How to cook Fried Egg;  Perfect Round;  every time |

## MakeOrangeJuice

23 matching videos. Steps: cut oranges → juice the oranges → pour the orange juice into the cup.

Good backup. 3 steps that look different (knife, juicer, cup). Videos mix hand juicers and blenders.

| Video | Split | Task length (s) | cov | Step times (s) | Title |
| --- | --- | --- | --- | --- | --- |
| [h930VZGBSRg](https://www.youtube.com/watch?v=h930VZGBSRg&t=2s) | training | 18 | 0.68 | 2–5, 6–11, 14–18 | DIY ORANGE JUICE || HOMEMADE DETOX JUICE || How to |
| [kwyH8D-h2zU](https://www.youtube.com/watch?v=kwyH8D-h2zU&t=35s) | training | 39 | 0.61 | 35–43, 44–55, 65–70 | Making Fresh Squeezed Orange Juice |
| [0siIuxPtRN4](https://www.youtube.com/watch?v=0siIuxPtRN4&t=24s) | training | 41 | 0.41 | 24–29, 37–42, 53–60 | Instant Orange juice, how to make fresh Orange jui |
| [C-W0LnTxosg](https://www.youtube.com/watch?v=C-W0LnTxosg&t=16s) | testing | 54 | 0.78 | 16–30, 32–58, 59–61 | Making Fresh Pressed Orange Juice! RAW! |
| [fJPGC7PXkDo](https://www.youtube.com/watch?v=fJPGC7PXkDo&t=8s) | training | 59 | 0.73 | 8–16, 23–54, 55–59 | How To Make Homemade Orange Juice Using A Blender |

## UseToaster

11 matching videos. Steps: put a slice of bread in → run the toaster and adjust → take out the slice of bread.

Easy to record but only 11 videos, and the set includes a panini press and a novelty toaster.

| Video | Split | Task length (s) | cov | Step times (s) | Title |
| --- | --- | --- | --- | --- | --- |
| [Pecoog5Gyfo](https://www.youtube.com/watch?v=Pecoog5Gyfo&t=34s) | training | 16 | 0.81 | 34–37, 38–43, 43–48 | Duo popup Pop Up Toaster |
| [uWwHeuHbuUk](https://www.youtube.com/watch?v=uWwHeuHbuUk&t=23s) | training | 24 | 0.78 | 23–26, 26–39, 40–43 | How to use a toaster [CC] |
| [VY4UJnOfeYg](https://www.youtube.com/watch?v=VY4UJnOfeYg&t=130s) | training | 31 | 0.57 | 130–138, 138–140, 148–157 | Cooking & Culinary Tips : How to Use a Panini Pres |
| [shmLNKXVGtA](https://www.youtube.com/watch?v=shmLNKXVGtA&t=20s) | training | 12 | 0.32 | 20–21, 21–23, 23–24 | Cello Pop up Toaster |
| [tH_fLWdxmV8](https://www.youtube.com/watch?v=tH_fLWdxmV8&t=2s) | training | 23 | 0.26 | 2–4, 5–6, 18–21 | Panda Toaster Panda Toaster |

## RefillAStapler

47 matching videos. Steps: pull open the stapler → insert the stapler pins → close up the stapler.

Easy and short, 47 videos, but the set mixes desk staplers with staple guns and a tab gun. Mistakes (staples upside down) are tiny on screen.

| Video | Split | Task length (s) | cov | Step times (s) | Title |
| --- | --- | --- | --- | --- | --- |
| [QpQ_OQdb_To](https://www.youtube.com/watch?v=QpQ_OQdb_To&t=7s) | training | 15 | 0.65 | 7–11, 12–15, 16–19 | PaperPro One Finger 15 Sheet Compact Stapler |
| [mj6CMZwFb3k](https://www.youtube.com/watch?v=mj6CMZwFb3k&t=7s) | training | 13 | 0.63 | 7–9, 10–13, 14–17 | Rapid Omnipress 60 Full Strip Stapler |
| [_KzroMEH90w](https://www.youtube.com/watch?v=_KzroMEH90w&t=34s) | training | 18 | 0.6 | 34–39, 40–43, 44–47 | dewalt staple gun |
| [HQs3l34setw](https://www.youtube.com/watch?v=HQs3l34setw&t=8s) | training | 9 | 0.69 | 8–11, 12–13, 14–16 | 資料裝訂夾非訂書機使用方法影片 |
| [vbsqiFUisTM](https://www.youtube.com/watch?v=vbsqiFUisTM&t=16s) | testing | 29 | 0.65 | 16–25, 26–31, 32–37 | Framers Corner Tab Gun Demo (PFATG) |

## ReplaceBatteryOnTVControl

13 matching videos. Steps: open cover → remove battery → put battery in → close cover.

Easy, 4 steps, but only 13 videos and a reversed battery is a few pixels on screen.

| Video | Split | Task length (s) | cov | Step times (s) | Title |
| --- | --- | --- | --- | --- | --- |
| [zcwYGVobmSI](https://www.youtube.com/watch?v=zcwYGVobmSI&t=36s) | training | 27 | 0.62 | 36–40, 42–48, 51–54, 55–60 | How To Change The Battery In An Apple Remote |
| [T0LtOOvnYq8](https://www.youtube.com/watch?v=T0LtOOvnYq8&t=8s) | training | 23 | 0.7 | 8–10, 12–16, 19–26, 26–30 | how to change battery in tv remote |
| [nk-PKZJP8JI](https://www.youtube.com/watch?v=nk-PKZJP8JI&t=14s) | training | 24 | 0.68 | 14–16, 17–22, 22–28, 29–32 | How to Change battery in Harmony Companion |
| [MS0SMLCCtIM](https://www.youtube.com/watch?v=MS0SMLCCtIM&t=7s) | testing | 17 | 0.49 | 7–8, 9–12, 14–16, 17–19 | Resetting the VUR10 Remote |
| [bIcw3oJUw-s](https://www.youtube.com/watch?v=bIcw3oJUw-s&t=8s) | training | 29 | 0.5 | 8–11, 18–21, 24–30, 30–34 | Control4 - Changing the battery on an SR-260 |

## Tasks with 4 or more steps (added 2026-10-05)

Counts are videos with exactly these steps, once each, in this order, out of all COIN videos for the task (on the mirror in brackets). "Likely mistakes" are errors a beginner makes for real and that a close camera can see.

| Task | Steps | Correct of annotated (mirror) | Likely mistakes |
| --- | --- | --- | --- |
| UseRiceCookerToCookRice | take out rice → soak and wash → put washed rice in cooker → cook | 31 of 101 (90) | skip washing; put rice in without water; forget to press cook |
| MakeFrenchFries | cut into strips → soak in water → dry → fry | 17 of 93 (77) | skip soaking; skip drying (wet strips into oil). Hot oil: use an air fryer or oven |
| PutOnQuiltCover | align quilt and cover → roll together → turn cover out from the other side → arrange | 26 of 40 (35) | cover inside out; corners not lined up; rolling from the wrong end |
| MakePaperWindMill | fold the edges → cut along the edges → fold squares inward and fix → fix on the stick | 25 of 106 (81) | cut too far to the centre; fold the wrong corners; pin through the wrong points |
| MakeRJ45Cable | strip → arrange wires → cut to length → insert into plug → crimp | 33 of 91 (86) | wrong colour order (the classic one); wires not fully in before crimping. Needs a ~$15 kit |
| ReplaceBatteryOnKeyToCar | open key cover → take out battery → put in battery → close cover | 34 of 46 (44) | battery upside down; cover not snapped shut. Small on screen; needs a close camera |

Also 4+ steps but less practical at home: ReplaceFilterForAirPurifier (53 of 99), ReplaceMemoryChip (51 of 95), ChangeTonerCartridge (48 of 92), ReplaceABulb (37 of 100). CookOmelet's 5-step version (bowl → stir → pan → fry → plate) has only 6 videos.
