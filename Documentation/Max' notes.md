## TOOLGROUPS 

Toolgroup wie zB. DE_BE_13 ist STNFAM in tool.txt
--> Familien müssen zwischen "<>" stehen

Area wie zB. Dry_etch ist STNGRP in tool.txt
--> Gruppen in "[]"

Man kann in der config.json auch mehrere "Mischungen" gleichzeitig verwenden, 
in der Form "station_group": ["[Dry_Etch]","<TE_BE_40>","[Implant]"]

Alle toolgroups wären dann also  :
["[Dry_Etch]","[Def_Met]","[Dielectric]","[Diffusion]","[Implant]","[Litho]","[Planar]","[TF]","[Wet_Etch]"]
