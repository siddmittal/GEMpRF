# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Medical University of Vienna",
"@Desc    :   Quick diagnostic viewer for BIDS-style folder structures.

               Rules used by this diagnostic tree:
               • Never dump the full directory structure - only a compact preview.
               • At each BIDS level (analysis → subject → session → files):
                    - show top 2 + "..." + bottom 2 items
                    - always expand the *last* matching analysis, subject, and session.
               • Stop early:
                    - if no matching analysis found → stop.
                    - if no matching subject inside that analysis → stop.
                    - if no matching session inside that subject → stop.
               • File listing:
                    - only consider fMRI-like files (.nii, .nii.gz, .gii, .mgz, .sifti, .dtseries.nii, .func.gii)
                    - list top 3 + "..." + bottom 3
                    - for each file, show why it was filtered (hemi/run/task/space/filetype).
               • Accept extended BIDS names:
                    - "ses-001nn", "sub-xyz-extra", or multiple hyphens are allowed.
               • Designed to explain *why* zero files matched without overwhelming the user.
",
        
"""

import os
from gem.utils.logger import Logger

class DiagnosticBidsTree:
    @classmethod
    def collect_bids_diagnostics(cls, base_path):
        diag = {
            "base_path": base_path,
            "analysis": [],
            "subjects": {},
            "sessions": {},
            "func_files": {},
        }

        if not os.path.isdir(base_path):
            diag["error"] = f"Base path does not exist: {base_path}"
            return diag

        # Level 1 → analysis-*
        analyses = sorted([d for d in os.listdir(base_path) if d.startswith("analysis-") and os.path.isdir(os.path.join(base_path, d))])
        diag["analysis"] = analyses

        for analysis in analyses:
            a_path = os.path.join(base_path, analysis)

            # Level 2 → sub-*
            subs = sorted([d for d in os.listdir(a_path) if d.startswith("sub-") and os.path.isdir(os.path.join(a_path, d))])
            diag["subjects"][analysis] = subs

            for sub in subs:
                s_path = os.path.join(a_path, sub)

                # Level 3 → ses-*
                sessions = sorted([d for d in os.listdir(s_path) if d.startswith("ses-") and os.path.isdir(os.path.join(s_path, d))])
                diag["sessions"][(analysis, sub)] = sessions

                for ses in sessions:
                    se_path = os.path.join(s_path, ses)

                    # Level 4 → func/*
                    func_path = os.path.join(se_path, "func")
                    if os.path.isdir(func_path):
                        try:
                            files = sorted(os.listdir(func_path))
                        except:
                            files = []
                    else:
                        files = []

                    diag["func_files"][(analysis, sub, ses)] = files

        return diag

    @classmethod
    def print_bids_diagnostic_tree(
        cls, diag,
        analysis_list, sub_list, ses_list, run_list, task_list, hemi_list, space_list, file_extension
    ):
        Logger.print_green_message("\n[ Diagnostic BIDS Tree - Quick Overview ]", print_file_name=False)

        if diag is None or "error" in diag:
            Logger.print_red_message("Diagnostic data could not be collected.\n", print_file_name=False)
            return

        # Shortening helper
        def ellip(items, top=2, bottom=2):
            if len(items) <= top + bottom:
                return items
            return items[:top] + ["..."] + items[-bottom:]

        # fMRI-like files whitelist
        fMRI_EXT = (".nii", ".nii.gz", ".gii", ".mgz", ".sifti", ".dtseries.nii", ".func.gii")

        # -------------------------
        # 1) ANALYSIS LEVEL
        # -------------------------
        analyses = diag["analysis"]
        print(f"📁 {diag.get('base_path', '')}")

        matched_analyses = [
            a for a in analyses
            if ('all' in analysis_list or a.split("analysis-")[-1] in analysis_list)
        ]

        # Stop if no matching analyses
        if not matched_analyses:
            print(" └── ❌ No analysis-* directories matching filters.")
            print(f"     Found: {', '.join(ellip(analyses))}\n")
            return

        # Only show matching analyses
        print(" ├── analysis:")
        ea = ellip(matched_analyses)
        for i, a in enumerate(ea):
            sym = "▾" if a == ea[-1] else "▸"
            print(f" │    ├── {sym} {a}")

        # Always expand last analysis
        last_analysis = matched_analyses[-1]

        # -------------------------
        # 2) SUBJECT LEVEL
        # -------------------------
        subjects = diag["subjects"].get(last_analysis, [])
        matched_subjects = [
            s for s in subjects
            if ('all' in sub_list or s.split("sub-")[-1] in sub_list)
        ]

        if not matched_subjects:
            print(f" │    └── ❌ No subjects matching filters in {last_analysis}")
            print(f"          Found: {', '.join(ellip(subjects))}\n")
            return

        print(f" │    Subjects in {last_analysis}:")
        es = ellip(matched_subjects)
        for i, s in enumerate(es):
            sym = "▾" if s == es[-1] else "▸"
            print(f" │        ├── {sym} {s}")

        # Always expand last subject
        last_subject = matched_subjects[-1]

        # -------------------------
        # 3) SESSION LEVEL
        # -------------------------
        sessions = diag["sessions"].get((last_analysis, last_subject), [])
        matched_sessions = [
            se for se in sessions
            if ('all' in ses_list or se.split("ses-")[-1] in ses_list)
        ]

        if not matched_sessions:
            print(f" │        └── ❌ No sessions matching filters for {last_subject}")
            print(f"              Found: {', '.join(ellip(sessions))}\n")
            return

        print(f" │        Sessions for {last_subject}:")
        eses = ellip(matched_sessions)
        for i, se in enumerate(eses):
            sym = "▾" if se == eses[-1] else "▸"
            print(f" │            ├── {sym} {se}")

        # Always expand last session
        last_session = matched_sessions[-1]

        # -------------------------
        # 4) FILE LEVEL
        # -------------------------
        files = diag["func_files"].get((last_analysis, last_subject, last_session), [])
        fmri_files = [f for f in files if f.endswith(fMRI_EXT)]

        # Apply filtering
        def file_filtered_reasons(f):
            r = []

            # extension
            if not (f.endswith(file_extension) or file_extension == "both"):
                r.append("filetype")

            # hemi
            if "hemi-" in f:
                hemi = f.split("hemi-")[1].split("_")[0]
                if not ("all" in hemi_list or hemi in hemi_list):
                    r.append(f"hemi={hemi}")

            # run
            if "run-" in f:
                run = f.split("run-")[1].split("_")[0]
                if not ("all" in run_list or run in run_list):
                    r.append(f"run={run}")

            # task
            if "task-" in f:
                task = f.split("task-")[1].split("_")[0]
                if task not in task_list:
                    r.append(f"task={task}")

            # space
            if "space-" in f:
                sp = f.split("space-")[1].split("_")[0]
                if not ("all" in space_list or sp in space_list):
                    r.append(f"space={sp}")

            return r

        # Gather diagnostic output
        filtered_files_output = []
        for f in fmri_files:
            reasons = file_filtered_reasons(f)
            filtered_files_output.append((f, reasons))

        if all(len(r) > 0 for _, r in filtered_files_output):
            print(f" │            ❌ No fMRI files matched filters in {last_session}")
            if fmri_files:
                print(" │            Found fMRI-like files:")
                for item in ellip(filtered_files_output, top=3, bottom=3):
                    if item == "...":
                        print(" │            ├── ...")
                        continue

                    f, reasons = item
                    print(f" │            ├── {f}   ✖ filtered: {', '.join(reasons)}")                    
            else:
                print(" │            Found: none")
            print("")
            return

        # # If this ever triggers, something is wrong
        # print(" │            Unexpected: some files passed filters but matching_files_info was empty.\n")

        # -----------------------------------------
        # NEW: If ANY file passed filters → show them
        # -----------------------------------------
        print(f" │            Matching fMRI files in {last_session}:")
        for f, reasons in filtered_files_output:
            if len(reasons) == 0:  # means this file passed all filters
                print(f" │            ├── ✔ {f}")
        print("")
        return        
        