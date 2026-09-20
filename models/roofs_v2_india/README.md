# Roof model v2 (India): D+ fine-tuned on Indian roofs

8-channel Mask R-CNN (RGB + our U-Net maps + teammate Inria/UAVid maps), continued from D+
(`maskrcnn_stacked8_dplus.pt`) for 2,000 steps on: Gandhinagar (the D+ training sector),
117 hand-labelled roofs from Jaipur / HSR / Dwarka / Chandigarh / Singh Nagar, and 48 crops of
UAVPal (Bhopal drone survey, 4,820 hand-drawn roofs; Maiti et al., IEEE JSTARS 2024,
doi 10.17026/dans-z55-6gt4, CC BY-NC-SA 4.0). Script: training/roofs_india/finetune_roofs.py.

| Held-out set | houses found (D+ -> v2) | building IoU | merged pairs |
|---|---|---|---|
| Indian hand-labelled (Jaipur, Singh Nagar, HSR) | 8/53 -> 27/53 | 0.52 -> 0.81 | 45% -> 26% |
| Bhopal east strip (UAVPal) | 63/1210 -> 485/1210 | 0.18 -> 0.68 | 11% -> 23% |
| Gandhinagar fair exam | 127/145 -> 129/145 | 0.88 -> 0.88 | 12.5% -> 9% |

Checked by eye (training/roofs_india/compare_roofs.py): most new "unmatched" shapes are real roofs
the labels draw coarser or miss; D+'s own false alarms on scrub/trees are gone. Known weak spots:
some large roofs split in two, wobbly outlines on big irregular buildings.
