Mutatex:
https://github.com/ELELAB/mutatex/blob/master/doc/manual.pdf 
Terminal instructions- ((mutatex-env) (protease-pipeline) [meiray@lgn12 mutatex]$ )

git clone https://github.com/ELELAB/mutatex.git
. mutatex-env/bin/activate
cd mutatex
cd  '/home/labs/rudich/meiray/meirab/foldx/mutatex'
python setup.py install
mutatex "/home/labs/rudich/meiray/meirab/foldx/1A0N.pdb" \ -m /home/labs/rudich/meiray/meirab/foldx/mutatex/templates/mutation_list.txt \ -f suite5 \ -R /home/labs/rudich/meiray/meirab/foldx/mutatex/templates/foldxsuite4/repair_runfile_template.txt \ -M /home/labs/rudich/meiray/meirab/foldx/mutatex/templates/foldxsuite4/mutate_runfile_template.txt \ -I /home/labs/rudich/meiray/meirab/foldx/mutatex/templates/foldxsuite4/interface_runfile_template.txt \ -x /home/labs/rudich/meiray/meirab/foldx/foldx_20241231 \ -b /home/labs/rudich/meiray/meirab/foldx/rotabase.txt
https://www.youtube.com/watch?v=4Dj4YbX2wjU&t=154s youtube tutorial of mutatex

