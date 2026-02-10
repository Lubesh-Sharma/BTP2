def save_pp_file(filename, VPos, indices):
    """
    Saves selected point indices to a MeshLab 'PickedPoints' (.pp) file format.
    """
    with open(filename, 'w') as f:
        f.write('<!DOCTYPE PickedPoints>\n<PickedPoints>\n')
        for i, idx in enumerate(indices):
            p = VPos[idx]
            f.write(f' <point x="{p[0]}" y="{p[1]}" z="{p[2]}" active="1" name="{i+1}"/>\n')
        f.write('</PickedPoints>\n')
