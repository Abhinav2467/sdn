# IEEE Manuscript Generation Skill

## Description
This skill automates the process of reviewing a project, generating results, and writing a professional IEEE manuscript in LaTeX. It includes a robust self-correction loop to iteratively refine the manuscript based on self-critiques.

## Instructions / Workflow

When invoked, strictly follow these steps:

1. **Project Review:** Thoroughly analyze the entire codebase, documentation, and project structure to fully understand the objectives, methodology, inputs, and expected outputs.
2. **Execution & Data Collection:** Run the necessary project scripts/code to generate and collect all required figures, plots, logs, and data tables needed for the manuscript. Ensure outputs are saved in an accessible directory.
3. **Content Drafting:** Draft the content for a professional IEEE format manuscript based on your project review and the collected data. Essential sections should include: Abstract, Introduction, Methodology, Results, Discussion, Conclusion, and References.
4. **LaTeX Document Creation:** Create the LaTeX source files using an IEEE template and integrate the drafted content and generated figures. Compile the document into a PDF using appropriate LaTeX commands (e.g., `pdflatex`, `bibtex`).
5. **Self-Critique & Error Tracking (Self-Loop):**
   - Critically review the generated PDF and LaTeX source for any errors, formatting issues, logical inconsistencies, or content weaknesses.
   - Summarize the critiques and write them concisely to an `errors_tracking.log` file. This file must be used to track all identified issues and subsequent changes.
6. **Issue Resolution:** Actively resolve the issues documented in the `errors_tracking.log` file by modifying the LaTeX source, refining the text, or even re-running project code if data needs correction.
7. **System Review:** After applying fixes, perform a comprehensive review of the updated system and manuscript to verify that the corrections were effective and that no new issues were introduced.
8. **Iterative Refinement:** 
   - **CRITICAL:** Every time a new critique cycle begins or an issue is detected, you MUST *first* review the `errors_tracking.log` file to understand historical mistakes and prevent regressions, *then* proceed with the current review and compilation.
   - Repeat steps 4 through 8 until the manuscript meets rigorous professional IEEE standards and the error log is cleared of outstanding issues.
