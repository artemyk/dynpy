make clean
mkdir ../htmldocs/html

cd ../htmldocs/html
git clone git@github.iu.edu:akolchin/DynPy.git .
git checkout origin/gh-pages -b gh-pages
git branch -d master

cd ../../docs

make html

echo "Created by following https://gist.github.com/chrisjacob/833223" > README.textile
git add .
git commit -m "Documentation updated"

git push origin gh-pages

cd ../../docs

