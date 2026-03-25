# Development

## Organization of work

All updates to NanoPSD should be organized using GitHub issues.

Issues help track what work is being done, why it is needed, and who is responsible. This allows contributors to coordinate development efficiently and avoid duplicated effort.

Work on NanoPSD may include:

- bug fixes
- feature additions
- improvements in nanoparticle detection algorithms
- documentation updates
- reproducibility improvements
- preparation of figures or examples for publications
- performance improvements
- testing on new microscopy datasets (SEM/TEM)

When possible, each logical unit of work should correspond to one GitHub issue.

Generally, new features should involve:

1. Implementing the feature
2. Testing the feature on example images
3. Updating documentation if needed

Each of these may be tracked using separate issues when the work is substantial.

---

## Creating issues

When creating a new issue:

1. Assign yourself if you plan to work on it
2. Choose an appropriate label
3. Write a clear and descriptive title
4. Provide a short explanation of the objective

Example issue title:

Improve robustness of scale bar detection

Example issue description:

Scale bar detection occasionally fails for low-contrast TEM images. Improve detection reliability and test on multiple example images.

---

### Labels

Issues may use the following label types:

**Type labels**

[bug]  
Code is not behaving as expected and needs to be fixed.

[documentation]  
Improvements or corrections to documentation, README, or comments.

[enhancement]  
New feature or improvement to existing functionality.

[question]  
Discussion about design decisions or implementation details.

[task]  
General development activity such as testing, benchmarking, or preparing figures.

**Status labels**

[working]  
Work is currently in progress.

[paused]  
Work is temporarily paused.

[blocked]  
Work depends on completion of another issue.

[under review]  
Pull request has been submitted and is under review.

[multiple steps]  
Feature requires several steps (implementation, testing, documentation).

---

## Creating issues for feature additions

Feature additions often involve multiple steps such as implementation, testing, and documentation.

For larger features, create a main issue describing the overall feature, and list subtasks using Markdown checkboxes:

- [ ] implement feature
- [ ] test on example datasets
- [ ] update documentation
- [ ] add usage example

Subtasks may optionally be tracked as separate issues.

Once all steps are completed, close the main issue.

---

## Beginning work on an issue

1. Add the **working** label to the issue
2. Create a feature branch

Ensure your local repository is up to date:

`git fetch`  
`git checkout main`  
`git pull`  

Create a new branch:

`git checkout -b short-description-of-change`  

Push branch to GitHub:

`git push -u origin short-description-of-change`  

### Branch naming convention

Use lowercase letters and hyphens.

Examples:

fix-scale-bar-detection  
improve-contour-filtering  
update-readme-installation  
add-batch-processing  

---

## Working on an issue

Make small commits as you work.

Try to ensure that the code runs correctly after each commit.

Example commit messages:

Fix scale bar detection threshold  
Improve contour filtering robustness  
Update documentation for batch processing  
Clean up deprecated comments  

After pushing your branch, you may open a pull request.

To close an issue automatically when the pull request is merged, include in the pull request description:

Closes #issue_number

Example:

Closes #15

---

## Pull request naming convention

Pull request titles should briefly describe the change.

Example:

PR: Improve robustness of particle detection

---

## Updating branch with latest main

If the main branch has been updated:

`git checkout main`  
`git pull`  
`git checkout branch-name`  
`git merge main`  

Resolve any conflicts if necessary.

---

## Sharing work for review

Push latest changes:

`git push`  

Create a pull request and request review.

Update issue labels:

remove [working]  
add [under review]  

Address reviewer comments and update the branch as needed.

Ensure:

- code runs correctly
- no errors occur
- outputs are generated as expected
- documentation is updated if necessary

---

## Merging approved pull requests

Before merging:

`git checkout main`  
`git pull`  
`git checkout branch-name`  
`git merge main`  
`git push`  

Confirm everything runs correctly.

Use "Squash and merge" when merging pull requests to keep commit history clean.

---

## Clean up after merge

Delete remote branch.

Delete local branch:

`git branch -D branch-name`  

Confirm related issue is closed.

---

## Code style guidelines

General principles:

- write clear and readable Python code
- use descriptive variable names
- include helpful comments where needed
- avoid unnecessary complexity
- maintain consistency with existing code structure

---

## Testing

When modifying NanoPSD:

- test using example microscopy images
- verify particle detection results visually
- confirm CSV output values are reasonable
- confirm generated plots appear correct

---

## Documentation

Update documentation when functionality changes.

Documentation locations may include:

README.md  
code comments  
usage examples  

---

## Questions

If unsure about design decisions, open an issue for discussion.

Contributions that improve clarity, usability, reproducibility, and scientific reliability are highly appreciated.