/*
SPDX-License-Identifier:    GPL-3.0-only
Copyright (C) 2026 Hippolyte Audet-Lagacé
Author:         Hippolyte Audet-Lagacé
Description:    This code create a Quadilateralized spherical cube (QSC) representing the earth, details are given for each functions.
                Note that in order to use read_elevations(), you need a valid file of the valid size (sphere subdivision) to read. As of the 
                last time this description was updated, there is on such file for 9 subdivision called elevations.bin .
*/

use std::fs::File;
use std::io::{BufWriter, Write, BufReader, Read};
use std::path::PathBuf;


#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub elevation: [f32; 3],
    pub rgba: [u8; 4],
}

// For explicitness
type Quaternion<T> = (T, [T; 3]);

/* -> Sphere -> look inside -> cube >:( */
pub struct Sphere {
    pub vertices: Box<[Vertex]>,
    pub indices: Box<[u32]>,
}
impl Sphere {
    /* 
    Builds a Quadrilateralized Spherical Cube (QSC).
    Each of the 6 cube faces is divided into (subdivision+1) by (subdivision+1) grids.
    The subdivision variable is a power of two (subdivison = 9 gives 512 for example).
    Since every future operations on the sphere in this program will be done with compute shaders, only new() is set to public
    */
    pub fn new(subdivision: u32) -> Result<Self,  Box<dyn std::error::Error>> {
        if subdivision > 15 {
            return Err("2 less subdivision is already too many vertices for the task at hand buddy".into());
        }
        if subdivision == 0 {
            return Err("The QSC equations break for 0 subdivisions, also you just tried to make a literal non curved cube".into());
        }

        // Initialization
        let n_vertices: usize = 2_usize.pow(subdivision).pow(2) * 6 + 2;  // Two corners are unnacounted for in the faces grid structure.
        let n_indices: usize = 2_usize.pow(subdivision).pow(2) * 36;      // 6 indices per rectangle, (2^subdivision)^2 rectangles, 6 faces.
        let mut vertices: Box<[Vertex]> = vec![Vertex::default(); n_vertices].into_boxed_slice();
        let mut indices: Box<[u32]> = vec![0; n_indices].into_boxed_slice();    // Vulkan uses u32 for indexing

        // Filling the struct
        make_sphere_vertices(&mut vertices, subdivision);
        make_sphere_indices(&mut indices, subdivision);
        
        return Ok(Sphere {vertices, indices});
    }

    /* Writes the positions data to filename file in little-endian byte order, use "name.bin" for filename */
    pub fn write_positions(&self, filename: &str) -> std::io::Result<()> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("auxiliaries");
        path.push("elevation");
        path.push(filename);

        println!("writing to: {}", path.display());

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        for v in self.vertices.iter() {
            for f in v.position {
                writer.write_all(&f.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }
    /* Reads elevation data from filename and assigns it to the elevation field of each vertex, preserving the same indexing order as the positions written by write_positions */
    pub fn read_elevations(&mut self, filename: &str) -> std::io::Result<()> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("auxiliaries");
        path.push("elevation");
        path.push(filename);
        println!("reading from: {}", path.display());

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        for v in self.vertices.iter_mut() {
            let mut elev = [0f32; 3];
            for f in elev.iter_mut() {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                *f = f32::from_le_bytes(buf);
            }
            v.elevation = elev;
        }

        Ok(())
    }
}

// Helper functions -----------------------------------------------------------------------------------------------------------

/* Generates the index array for the index buffer of a vertex buffer built with the associated vertex array. */
fn make_sphere_indices(ind: &mut Box<[u32]>, subdivision: u32) {
    let edge_size: usize = 2_usize.pow(subdivision);
    let offset: usize = (2 * subdivision) as usize;

    // Treating the faces one by one, with their right and top gap to the next face except for faces 4 (no gap around )and 5 (gaps on all sides)
    // Face 0
    let mut face: usize = 0;
    let mut current_y: usize = 0; // variable used for out-of-loop y-values
    let mut current_vertex: usize = 0;
    let mut neighbors: [usize; 4] = [0,0,0,0];
    let (mut right, mut top) = (neighbors[0], neighbors[1]);
    let mut topleft = top;
    for y in 0..(edge_size - 1) {
        // Row initialization and first triangle
        current_vertex = z_order_encode(0, y.try_into().unwrap());
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[1]);
        (ind[6_usize * y * edge_size], ind[6_usize * y * edge_size + 1_usize], ind[6_usize * y * edge_size + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
        // Fill the inner triangles of the row
        for x in 1..edge_size {
            current_vertex = right;
            neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
            (right, top) = (neighbors[0], neighbors[1]);
            (ind[6_usize * y * edge_size + 6_usize * x - 3_usize], ind[6_usize * y * edge_size + 6_usize * x - 2_usize], ind[6_usize * y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
            (ind[6_usize * y * edge_size + 6_usize * x], ind[6_usize * y * edge_size + 6_usize * x + 1_usize], ind[6_usize * y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
            topleft = top;
        }
        // Last triangle
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        top = neighbors[1];
        (ind[6_usize * (y + 1_usize) * edge_size - 3_usize], ind[6_usize * (y + 1_usize) * edge_size - 2_usize], ind[6_usize * (y + 1_usize) * edge_size - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    }
    // top row without last rectangle (current_y = edge_size - 1 ; last y-position of the face)
    current_y = edge_size - 1;
    current_vertex = face + z_order_encode(0, current_y.try_into().unwrap());
    neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
    (right, top) = (neighbors[0], neighbors[1]);
    (ind[6_usize * current_y * edge_size], ind[6_usize * current_y * edge_size + 1_usize], ind[6_usize * current_y * edge_size + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
    topleft = top;
    // Fill the inner triangles of the row
    for x in 1..(edge_size - 1) {
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[1]);
        (ind[6_usize * current_y * edge_size + 6_usize * x - 3_usize], ind[6_usize * current_y * edge_size + 6_usize * x - 2_usize], ind[6_usize * current_y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
        (ind[6_usize * current_y * edge_size + 6_usize * x + 0_usize], ind[6_usize * current_y * edge_size + 6_usize * x + 1_usize], ind[6_usize * current_y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
    }
    // Last triangle
    current_vertex = right;
    neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
    top = neighbors[1];
    (ind[6_usize * (current_y + 1_usize) * edge_size - 9_usize], ind[6_usize * (current_y + 1_usize) * edge_size - 8_usize], ind[6_usize * (current_y + 1_usize) * edge_size - 7_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    // Extra square near north
    ind[6_usize * (current_y + 1_usize) * edge_size - 6_usize] = (6_usize << offset) as u32;
    ind[6_usize * (current_y + 1_usize) * edge_size - 5_usize] = (z_order_encode(0, (edge_size - 1).try_into().unwrap())) as u32;
    ind[6_usize * (current_y + 1_usize) * edge_size - 4_usize] = ((4_usize << offset) + z_order_encode(0, (edge_size - 1).try_into().unwrap())) as u32;
    ind[6_usize * (current_y + 1_usize) * edge_size - 3_usize] = ((4_usize << offset) + z_order_encode(0, (edge_size - 1).try_into().unwrap())) as u32;
    ind[6_usize * (current_y + 1_usize) * edge_size - 2_usize] = (z_order_encode(0, (edge_size - 1).try_into().unwrap())) as u32;
    ind[6_usize * (current_y + 1_usize) * edge_size - 1_usize] = ((4_usize << offset) + z_order_encode(0, (edge_size - 2).try_into().unwrap())) as u32;
    // Face 1
    face = 1_usize << offset;
    for y in 0..edge_size {
        // Row initialization and first triangle
        current_vertex = face + z_order_encode(0, y.try_into().unwrap());
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[1]);
        (ind[6_usize * face + 6_usize * y * edge_size], ind[6_usize * face + 6_usize * y * edge_size + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
        // Fill the inner triangles of the row
        for x in 1..edge_size {
            current_vertex = right;
            neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
            (right, top) = (neighbors[0], neighbors[1]);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 3_usize], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) - 2_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
            topleft = top;
        }
        // Last triangle
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        top = neighbors[1];
        (ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 3_usize], ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 2_usize], ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    }
       
    // Face 2 
    face = 2_usize << offset;
    for y in 0..(edge_size - 1) {
        // Row initialization and first triangle
        current_vertex = face + z_order_encode(0, y.try_into().unwrap());
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[1]);
        (ind[6_usize * face + 6_usize * y * edge_size], ind[6_usize * face + 6_usize * y * edge_size + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
        // Fill the inner triangles of the row
        for x in 1..edge_size {
            current_vertex = right;
            neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
            (right, top) = (neighbors[0], neighbors[1]);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 3_usize], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) - 2_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
            topleft = top;
        }
        // Last triangle
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        top = neighbors[1];
        (ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 3_usize], ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 2_usize], ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    }
    // top row without last rectangle
    current_y = edge_size - 1;
    current_vertex = face + z_order_encode(0, current_y.try_into().unwrap());
    neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
    (right, top) = (neighbors[0], neighbors[1]);
    (ind[6_usize * face + 6_usize * current_y * edge_size], ind[6_usize * face + 6_usize * current_y * edge_size + 1_usize], ind[6_usize * face + 6_usize * current_y * edge_size + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
    topleft = top;
    // Fill the inner triangles of the row
    for x in 1..(edge_size - 1) {
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[1]);
        (ind[6_usize * face + 6_usize * current_y * edge_size + 6_usize * x - 3_usize], ind[(6_usize * face + 6_usize * current_y * edge_size) + (6_usize * x) - 2_usize], ind[6_usize * face + 6_usize * current_y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
        (ind[6_usize * face + 6_usize * current_y * edge_size + 6_usize * x], ind[(6_usize * face + 6_usize * current_y * edge_size) + (6_usize * x) + 1_usize], ind[6_usize * face + 6_usize * current_y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
    }
    // Last triangle
    current_vertex = right;
    neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
    top = neighbors[1];
    (ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 9_usize], ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 8_usize], ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 7_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    // Extra square near south
    ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 6_usize] = ((6_usize << offset) + 1) as u32;
    ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 5_usize] = (1_usize << offset) as u32;
    ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 4_usize] = ((2_usize << offset) + z_order_encode((edge_size - 1).try_into().unwrap(), 0)) as u32;
    ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 3_usize] = ((2_usize << offset) + z_order_encode((edge_size - 1).try_into().unwrap(), 0)) as u32;
    ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 2_usize] = (1_usize << offset) as u32;
    ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 1_usize] = ((1_usize << offset) + z_order_encode(0, 1)) as u32;
        
    // Face 3
    face = 3_usize << offset;
    for y in 0..edge_size {
        // Row initialization and first triangle
        current_vertex = face + z_order_encode(0, y.try_into().unwrap());
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[1]);
        (ind[6_usize * face + 6_usize * y * edge_size], ind[6_usize * face + 6_usize * y * edge_size + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
        // Fill the inner triangles of the row
        for x in 1..edge_size {
            current_vertex = right;
            neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
            (right, top) = (neighbors[0], neighbors[1]);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 3_usize], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) - 2_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
            topleft = top;
        }
        // Last triangle
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        if y == 0 {
            top = neighbors[2]; // current is now in face 4, sideways relative to face 3
        } else {
            top = neighbors[1];
        }
        (ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 3_usize], ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 2_usize], ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    }
        
    // Face 4 >:( 
    face = 4_usize << offset;
    for y in 0..(edge_size - 1) {
        // Row initialization and first triangle
        current_vertex = face + z_order_encode(0, y.try_into().unwrap());
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[1]);
        (ind[6_usize * face + 6_usize * y * edge_size], ind[6_usize * face + 6_usize * y * edge_size + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
        // Fill the inner triangles of the row
        for x in 1..(edge_size - 1) {
            current_vertex = right;
            neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
            (right, top) = (neighbors[0], neighbors[1]);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 3_usize], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) - 2_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x + 0_usize], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
            topleft = top;
        }
        // Last triangle
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        top = neighbors[1];
        (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * (edge_size - 1_usize) - 3_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * (edge_size - 1_usize) - 2_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * (edge_size - 1_usize) - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    }
    // extra lines left and bottom of face 5 (stored in the index space of face 4 because this is where theres space remaining for it)
    // left stored in the gaps of the array space
    current_vertex = 5_usize << offset;
    neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
    (right, top) = (neighbors[1], neighbors[2]); // going upwards
    (ind[6_usize * face + 6_usize * (edge_size - 1_usize)], ind[6_usize * face + 6_usize * (edge_size - 1_usize) + 1_usize], ind[6_usize * face + 6_usize * (edge_size - 1_usize) + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
    topleft = top;
    for x in 1..(edge_size - 1) {
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[1], neighbors[2]);
        (ind[6_usize * face + 6_usize * edge_size * x - 3_usize], ind[6_usize * face + 6_usize * edge_size * x - 2_usize], ind[6_usize * face + 6_usize * edge_size * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
        (ind[6_usize * face + 6_usize * edge_size * (x + 1_usize) - 6_usize], ind[6_usize * face + 6_usize * edge_size * (x + 1_usize)  - 5_usize], ind[6_usize * face + 6_usize * edge_size * (x + 1_usize)  - 4_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
    }
    current_vertex = right;
    neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
    top = neighbors[2];
    (ind[6_usize * face + 6_usize * (edge_size - 1_usize) * edge_size - 3_usize], ind[6_usize * face + 6_usize * (edge_size - 1_usize) * edge_size - 2_usize], ind[6_usize * face + 6_usize * (edge_size - 1_usize) * edge_size - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    // bottom stored at the end of the face's array space
    current_y = edge_size - 1;
    current_vertex = 5_usize << offset;
    neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
    (right, top) = (neighbors[0], neighbors[3]); // upside down so we do clockwise winding for this stretch
    (ind[6_usize * face + 6_usize * current_y * edge_size], ind[6_usize * face + 6_usize * current_y * edge_size + 1_usize], ind[6_usize * face + 6_usize * current_y * edge_size + 2_usize]) = (current_vertex as u32, top as u32, right as u32);
    topleft = top;
    for x in 1..edge_size {
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[3]);
        (ind[6_usize * face + 6_usize * current_y * edge_size + 6_usize * x - 3_usize], ind[(6_usize * face + 6_usize * current_y * edge_size) + (6_usize * x) - 2_usize], ind[6_usize * face + 6_usize * current_y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, top as u32, current_vertex as u32);
        (ind[6_usize * face + 6_usize * current_y * edge_size + 6_usize * x], ind[(6_usize * face + 6_usize * current_y * edge_size) + (6_usize * x) + 1_usize], ind[6_usize * face + 6_usize * current_y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, top as u32, right as u32);
        topleft = top;
    }
    current_vertex = right;
    neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
    top = neighbors[0];
    (ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 3_usize], ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 2_usize], ind[6_usize * face + 6_usize * (current_y + 1_usize) * edge_size - 1_usize]) = (topleft as u32, top as u32, current_vertex as u32);
        
    // Face 5
    face = 5_usize << offset;
    for y in 0..edge_size {
        // Row initialization and first triangle
        current_vertex = face + z_order_encode(0, y.try_into().unwrap());
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        (right, top) = (neighbors[0], neighbors[1]);
        (ind[6_usize * face + 6_usize * y * edge_size], ind[6_usize * face + 6_usize * y * edge_size + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
        topleft = top;
        // Fill the inner triangles of the row
        for x in 1..edge_size {
            current_vertex = right;
            neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
            (right, top) = (neighbors[0], neighbors[1]);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 3_usize], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) - 2_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
            (ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x + 0_usize], ind[(6_usize * face + 6_usize * y * edge_size) + (6_usize * x) + 1_usize], ind[6_usize * face + 6_usize * y * edge_size + 6_usize * x + 2_usize]) = (current_vertex as u32, right as u32, top as u32);
            topleft = top;
        }
        // Last triangle
        current_vertex = right;
        neighbors = find_nearest_neighbors(current_vertex, subdivision).unwrap();
        if y == edge_size - 1 {
            top = neighbors[1];
        } else {
            top = neighbors[2];
        }
        (ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 3_usize], ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 2_usize], ind[6_usize * face + 6_usize * (y + 1_usize) * edge_size - 1_usize]) = (topleft as u32, current_vertex as u32, top as u32);
    }
}
/* Generates the vertices of the QSC using slerp, with quad-tree indexing. 
Since the vertices need a position, z = 1 is assigned to the north pole and the local x = 0 edge of face 0 is in the x-z plane. */
fn make_sphere_vertices(vert: &mut Box<[Vertex]>, subdivision: u32) {
    // Two edges will be ignored per faces, giving a power of 2 as grid dimension
    let n: usize = 2_usize.pow(subdivision) + 1; 
    let pi: f32 = 3.14159265358979323846264338327950288;
    let theta_n = (1.0_f32 / 3.0_f32).acos();
    let theta_s = pi - theta_n;

    // bl, br, tl, tr: bottom left, ...
    // Face 0, owns it's top and right edges
    let mut face_id: usize = 0b000;
    let mut bl: [f32; 3] = spherical_to_cartesian(theta_n, 0.0);
    let mut br: [f32; 3] = spherical_to_cartesian(theta_s, pi/3.0);
    let mut tl: [f32; 3] = spherical_to_cartesian(0.0, pi/3.0);
    let mut tr: [f32; 3] = spherical_to_cartesian(theta_n, 2.0*pi/3.0);
    let mut grid: Box<[Box<[[f32;3]]>]> = build_grid(bl, br, tl, tr, n);
    build_face_vertices(vert, subdivision as usize, face_id, grid, 1, n, 1, n);

    // Face 1, owns it's bottom and right edges
    face_id = 0b001;
    bl = spherical_to_cartesian(pi, 4.0*pi/3.0);
    br = spherical_to_cartesian(theta_s, 5.0*pi/3.0);
    tl = spherical_to_cartesian(theta_s, pi);
    tr = spherical_to_cartesian(theta_n, 4.0*pi/3.0);
    grid = build_grid(bl, br, tl, tr, n);
    build_face_vertices(vert, subdivision as usize, face_id, grid, 1, n, 0, n-1);

    // Face 2, owns it's top and right edges
    face_id = 0b010;
    bl = spherical_to_cartesian(theta_s, pi/3.0);
    br = spherical_to_cartesian(pi, 4.0*pi/3.0);
    tl = spherical_to_cartesian(theta_n, 2.0*pi/3.0);
    tr = spherical_to_cartesian(theta_s, pi);
    grid = build_grid(bl, br, tl, tr, n);
    build_face_vertices(vert, subdivision as usize, face_id, grid, 1, n, 1, n);

    // Face 3, owns it's bottom and right edges
    face_id = 0b011;
    bl = spherical_to_cartesian(theta_s, 5.0*pi/3.0);
    br = spherical_to_cartesian(theta_n, 0.0);
    tl = spherical_to_cartesian(theta_n, 4.0*pi/3.0);
    tr = spherical_to_cartesian(0.0, pi/3.0);
    grid = build_grid(bl, br, tl, tr, n);
    build_face_vertices(vert, subdivision as usize, face_id, grid, 1, n, 0, n-1);

    // Face 4, owns it's top and right edges
    face_id = 0b100;
    bl = spherical_to_cartesian(theta_n, 2.0*pi/3.0);
    br = spherical_to_cartesian(theta_s, pi);
    tl = spherical_to_cartesian(0.0, pi/3.0);
    tr = spherical_to_cartesian(theta_n, 4.0*pi/3.0);
    grid = build_grid(bl, br, tl, tr, n);
    build_face_vertices(vert, subdivision as usize, face_id, grid, 1, n, 1, n);

    // Face 5, owns it's top and left edges
    face_id = 0b101;
    bl = spherical_to_cartesian(theta_n, 0.0);
    br = spherical_to_cartesian(theta_s, 5.0*pi/3.0);
    tl = spherical_to_cartesian(theta_s, pi/3.0);
    tr = spherical_to_cartesian(pi, 4.0*pi/3.0);
    grid = build_grid(bl, br, tl, tr, n);
    build_face_vertices(vert, subdivision as usize, face_id, grid, 0, n-1, 1, n);

    // Poles 
    let north: [f32; 3] = [0.0, 0.0, 1.0];
    let south: [f32; 3] = [0.0, 0.0, -1.0];
    vert[2_usize.pow(subdivision).pow(2) * 6_usize].position = north;
    vert[2_usize.pow(subdivision).pow(2) * 6_usize + 1_usize].position = south;
}
/* For every vertices in a face, compute the cartesian coordinate and the quad-tree index */
fn build_face_vertices(
    vert: &mut Box<[Vertex]>, 
    subdivision: usize,
    face_id: usize,
    grid:  Box<[Box<[[f32;3]]>]>,
    start_x: usize,
    end_x: usize,
    start_y: usize,
    end_y: usize,
) {
    for y in start_y..end_y {
        for x in start_x..end_x {
            vert[(face_id << 2 * subdivision) + z_order_encode((x - start_x) as u32, (y - start_y) as u32)].position = grid[y][x];
            if face_id == 0b000 { vert[(face_id << 2 * subdivision) + z_order_encode((x - start_x) as u32, (y - start_y) as u32)].rgba[0] = 170 };
            if face_id == 0b001 { vert[(face_id << 2 * subdivision) + z_order_encode((x - start_x) as u32, (y - start_y) as u32)].rgba[0] = 170 };
            if face_id == 0b010 { vert[(face_id << 2 * subdivision) + z_order_encode((x - start_x) as u32, (y - start_y) as u32)].rgba[1] = 170 };
            if face_id == 0b011 { vert[(face_id << 2 * subdivision) + z_order_encode((x - start_x) as u32, (y - start_y) as u32)].rgba[1] = 170 };
            if face_id == 0b100 { vert[(face_id << 2 * subdivision) + z_order_encode((x - start_x) as u32, (y - start_y) as u32)].rgba[2] = 170 };
            if face_id == 0b101 { vert[(face_id << 2 * subdivision) + z_order_encode((x - start_x) as u32, (y - start_y) as u32)].rgba[2] = 170 };
        }
    }
}
/* Builds a grid on planar space using slerp, returns the grid in format Box[y][x] */
fn build_grid(bl: [f32; 3], br: [f32; 3], tl: [f32; 3], tr: [f32; 3], size: usize) -> Box<[Box<[[f32;3]]>]> {
    let mut result:  Box<[Box<[[f32;3]]>]> =  vec![vec![[0.0, 0.0, 0.0]; size].into_boxed_slice(); size].into_boxed_slice();
    let steps: f32 = size as f32 - 1.0;

    let row_starts: Box<[[f32; 3]]> = slerp(build_partial_rotation(bl, tl, steps), &bl, size);
    let row_ends: Box<[[f32; 3]]> = slerp(build_partial_rotation(br, tr, steps), &br, size);

    for i in 0..size {
        result[i] = slerp(build_partial_rotation(row_starts[i], row_ends[i], steps), &row_starts[i], size);
    }

    return result;
}
/* Builds a rotation quaternion covering 1/steps of the path from u to v */
fn build_partial_rotation(u: [f32; 3], v: [f32; 3], steps: f32) -> Quaternion<f32> {
    let angle: f32 = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]).acos();
    let rotation: f32 = angle / (2.0 * steps);
    let sin_angle: f32 = angle.sin();
    let axis: [f32; 3] = [
        (u[1]*v[2] - u[2]*v[1]) / sin_angle * rotation.sin(),
        (u[2]*v[0] - u[0]*v[2]) / sin_angle * rotation.sin(),
        (u[0]*v[1] - u[1]*v[0]) / sin_angle * rotation.sin(),
    ];
    
    return (rotation.cos() , axis);
}
/* Rotate a vector by a given Quaternion */
fn rotate_by_quaternion(q: Quaternion<f32>, u: &mut [f32; 3]) {
    let (s, v) = q;

    let a: [f32; 3] = v.map(|x| x * 2.0_f32 * (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]));
    let b: [f32; 3] = u.map(|x| x * (s * s - (v[0]*v[0] + v[1]*v[1] + v[2]*v[2])));
    let cross_v_u: [f32; 3] = [v[1]*u[2] - v[2]*u[1], v[2]*u[0] - v[0]*u[2], v[0]*u[1] - v[1]*u[0]];
    let c: [f32; 3] = cross_v_u.map(|x| x *2.0_f32 * s);

    for i in 0..3 {
        u[i] = a[i] + b[i] + c[i];
    }
}
/* Spherical linear interpolation using a prebuild rotation quaternion */
fn slerp(q: Quaternion<f32>, u: &[f32; 3], repeats: usize) -> Box<[[f32; 3]]> {
    let mut result: Box<[[f32; 3]]> = vec![[0.0, 0.0, 0.0]; repeats].into_boxed_slice();
    let mut v: [f32; 3] = *u;

    for i in 0..repeats {
        result[i] = v;
        rotate_by_quaternion(q, &mut v);
    }

    return result;
}
/* performs a Z-order encoding or decoding operation */
fn z_order_encode(x: u32, y: u32) -> usize {
    fn part(n: u32) -> u64 {
        let mut n = n as u64;
        n &= 0x00000000ffffffff;
        n = (n | (n << 16)) & 0x0000ffff0000ffff;
        n = (n | (n << 8))  & 0x00ff00ff00ff00ff;
        n = (n | (n << 4))  & 0x0f0f0f0f0f0f0f0f;
        n = (n | (n << 2))  & 0x3333333333333333;
        n = (n | (n << 1))  & 0x5555555555555555;
        return n;
    }
    return (part(x) | (part(y) << 1)) as usize;
}
fn z_order_decode(z: usize) -> (u32, u32) {
    fn compact(n: usize) -> u32 {
        let mut n = n & 0x5555555555555555;
        n = (n | (n >> 1))  & 0x3333333333333333;
        n = (n | (n >> 2))  & 0x0f0f0f0f0f0f0f0f;
        n = (n | (n >> 4))  & 0x00ff00ff00ff00ff;
        n = (n | (n >> 8))  & 0x0000ffff0000ffff;
        n = (n | (n >> 16)) & 0x00000000ffffffff;
        return n as u32;
    }
    return (compact(z), compact(z >> 1));
}
/* Converts spherical coordinates from a unit sphere to cartesian within [(-1,-1,-1), (1,1,1)] */ 
fn spherical_to_cartesian(phi: f32, theta: f32) -> [f32; 3] {
    let l = phi.sin() * theta.cos();
    let m = phi.sin() * theta.sin();
    let n = phi.cos();
    return [l, m, n];
}
/* Find the 4 nearest neighbors [right, top, left, bottom] index of a given index assuming the indexing is the one found in the Sphere struct
Corners have only 3 neighbors and as such the index of the owned corner of face  (0:right, 1:bottom, 2:right, 3:bottom, 4:right, 5:top) neighbor will be an error index (face 6 index 2) or a duplicate */
fn find_nearest_neighbors(vert_index: usize, subdivision: u32) -> Result<[usize; 4], Box<dyn std::error::Error>> {
    if subdivision >= usize::BITS {
        return Err("You can't have a sphere this large and you shouldn't even have reached this message".into())
    }

    // Isolating relevant parameters
    let offset: usize = (2 * subdivision) as usize;
    let mut face_id: usize = (vert_index >> offset) & 0b111;
    let in_face_index: usize = vert_index & ((1 << offset) - 1);
    let (x, y): (u32, u32) = z_order_decode(in_face_index);
    let local_max: u32 = 2_u32.pow(subdivision) - 1; 

    // Edges and corners lookup table (neighbors are [right, top, left, bottom])
    match face_id {
        0 => match (x, y) {
            (0, 0) => return Ok([z_order_encode(1, 0), z_order_encode(0, 1), (3 << offset) + z_order_encode(local_max, local_max - 1), (5 << offset) + 0]),
            (0, b) if b == local_max => return  Ok([z_order_encode(1, local_max), (4 << offset) +  z_order_encode(0, local_max - 1), (6 << offset), z_order_encode(0, local_max - 1)]),
            (a, 0) if a == local_max => return  Ok([(2 << offset) + 0, z_order_encode(local_max, 1), z_order_encode(local_max - 1, 0), (5 << offset) + z_order_encode(0, local_max)]),
            (a, b) if a == local_max && b == local_max => return  Ok([(6 << offset) + 2, (2 << offset) + z_order_encode(0, local_max), z_order_encode(local_max - 1, local_max), z_order_encode(local_max, local_max - 1)]),
            (0, _) => return  Ok([z_order_encode(1, y), z_order_encode(0, y + 1), (3 << offset) + z_order_encode(local_max, y + 1), z_order_encode(0, y - 1)]),
            (a, _) if a == local_max => return  Ok([(2 << offset) + z_order_encode(0, y), z_order_encode(local_max, y + 1), z_order_encode(local_max - 1, y), z_order_encode(local_max, y - 1)]),
            (_, 0) => return  Ok([z_order_encode(x + 1, 0), z_order_encode(x, 1), z_order_encode(x - 1, 0), (5 << offset) + z_order_encode(0, x)]),
            (_, b) if b == local_max => return  Ok([z_order_encode(x + 1, local_max), (4 << offset) + z_order_encode(0, local_max - x - 1), z_order_encode(x - 1, local_max), z_order_encode(x, local_max - 1)]),
            _ => {}
        },
        1 => match (x, y) {
            (0, 0) => return  Ok([(1 << offset) + z_order_encode(1, 0),(1 << offset) + z_order_encode(0, 1) ,(6 << offset) + 1, (5 << offset) + z_order_encode(local_max, local_max - 1)]),
            (0, b) if b == local_max => return  Ok([(1 << offset) + z_order_encode(1, local_max), (4 << offset) + z_order_encode(local_max, 0),(2 << offset) + z_order_encode(local_max, y - 1), (1 << offset) + z_order_encode(0, local_max - 1)]),
            (a, 0) if a == local_max => return  Ok([(3 << offset) + z_order_encode(0, 0), (1 << offset) + z_order_encode(local_max, 1), (1 << offset) + z_order_encode(local_max - 1, 0), (6 << offset) + 2]),
            (a, b) if a == local_max && b == local_max => return  Ok([(3 << offset) + z_order_encode(0, local_max), (4 << offset) + z_order_encode(local_max, local_max), (1 << offset) + z_order_encode(local_max - 1, local_max), (1 << offset) + z_order_encode(local_max, local_max - 1)]),
            (0, _) => return  Ok([(1 << offset) + z_order_encode(1, y), (1 << offset) + z_order_encode(0, y + 1), (2 << offset) + z_order_encode(local_max, y - 1), (1 << offset) + z_order_encode(0, y - 1)]),
            (a, _) if a == local_max => return  Ok([(3 << offset) +z_order_encode(0, y), (1 << offset) + z_order_encode(local_max, y + 1), (1 << offset) + z_order_encode(local_max - 1, y), (1 << offset) + z_order_encode(local_max, y - 1)]),
            (_, 0) => return  Ok([(1 << offset) + z_order_encode(x + 1, 0), (1 << offset) + z_order_encode(x, 1), (1 << offset) + z_order_encode(x - 1, 0), (5 << offset) + z_order_encode(local_max, local_max - x - 1)]),
            (_, b) if b == local_max => return  Ok([(1 << offset) + z_order_encode(x + 1, local_max), (4 << offset) + z_order_encode(local_max, x), (1 << offset) + z_order_encode(x - 1, local_max), (1 << offset) + z_order_encode(x, local_max - 1)]),
            _ => {}
        },
        2 => match (x, y) {
            (0, 0) => return  Ok([(2 << offset) + z_order_encode(1, 0), (2 << offset) + z_order_encode(0, 1), z_order_encode(local_max, 0), (5 << offset) + z_order_encode(1, local_max)]),
            (0, b) if b == local_max => return  Ok([(2 << offset) + z_order_encode(1, local_max), (4 << offset), z_order_encode(local_max, local_max), (2 << offset) + z_order_encode(0,local_max - 1)]),
            (a, 0) if a == local_max => return  Ok([(1 << offset) + z_order_encode(0, 1), (2 << offset) + z_order_encode(local_max, 1), (2 << offset) + z_order_encode(local_max - 1, 0), (6 << offset) + 1]),
            (a, b) if a == local_max && b == local_max => return  Ok([(6 << offset) + 2, (4 << offset) + z_order_encode(local_max, 0), (2 << offset) + z_order_encode(local_max - 1, local_max), (2 << offset) + z_order_encode(local_max, local_max - 1)]),
            (0, _) => return  Ok([(2 << offset) + z_order_encode(1, y), (2 << offset) + z_order_encode(0, y + 1), z_order_encode(local_max, y), (2 << offset) + z_order_encode(0, y - 1)]),
            (a, _) if a == local_max => return  Ok([(1 << offset) + z_order_encode(0, y + 1), (2 << offset) + z_order_encode(local_max, y + 1), (2 << offset) + z_order_encode(local_max - 1, y), (2 << offset) + z_order_encode(local_max, y - 1)]),
            (_, 0) => return  Ok([(2 << offset) + z_order_encode(x + 1, 0), (2 << offset) + z_order_encode(x, 1), (2 << offset) + z_order_encode(x - 1, 0), (5 << offset) + z_order_encode(x - 1, local_max)]),
            (_, b) if b == local_max => return  Ok([(2 << offset) + z_order_encode(x + 1, local_max), (4 << offset) + z_order_encode(x, 0), (2 << offset) + z_order_encode(x - 1, local_max), (2 << offset) + z_order_encode(x, local_max - 1)]),
            _ => {}
        },
        3 => match (x, y) {
            (0, 0) => return  Ok([(3 << offset) + z_order_encode(1, 0), (3 << offset) + z_order_encode(0, 1), (1 << offset) + z_order_encode(local_max, 0), (5 << offset) + z_order_encode(local_max, 0)]),
            (0, b) if b == local_max => return  Ok([(3 << offset) + z_order_encode(1, local_max), (4 << offset) + z_order_encode(local_max - 1, local_max), (1 << offset) + z_order_encode(local_max, local_max), (3 << offset) + z_order_encode(0, local_max - 1)]),
            (a, 0) if a == local_max => return  Ok([(5 << offset), (3 << offset) + z_order_encode(local_max, 1), (3 << offset) + z_order_encode(local_max - 1, 0), (6 << offset) + 2]),
            (a, b) if a == local_max && b == local_max => return  Ok([z_order_encode(0, y - 1), (6 << offset), (3 << offset) + z_order_encode(local_max - 1, local_max), (3 << offset) + z_order_encode(local_max - 1, local_max)]),
            (0, _) => return  Ok([(3 << offset) + z_order_encode(1, y), (3 << offset) + z_order_encode(0, y + 1), (1 << offset) + z_order_encode(local_max, y), (3 << offset) + z_order_encode(0, y - 1)]),
            (a, _) if a == local_max => return Ok([z_order_encode(0, y - 1), (3 << offset) + z_order_encode(local_max, y + 1), (3 << offset) + z_order_encode(local_max - 1, y), (3 << offset) + z_order_encode(local_max, y - 1)]),
            (_, 0) => return  Ok([(3 << offset) + z_order_encode(x + 1, 0), (3 << offset) + z_order_encode(x, 1), (3 << offset) + z_order_encode(x - 1, 0), (5 << offset) + z_order_encode(local_max - x, 0)]),
            (_, b) if b == local_max => return  Ok([(3 << offset) + z_order_encode(x + 1, local_max), (4 << offset) + z_order_encode(local_max - x - 1, local_max), (3 << offset) + z_order_encode(x - 1, local_max), (3 << offset) + z_order_encode(x, local_max - 1)]),
            _ => {}
        },
        4 => match (x, y) {
            (0, 0) => return  Ok([(4 << offset) + z_order_encode(1, 0), (4 << offset) + z_order_encode(0, 1), z_order_encode(local_max - 1, local_max), (2 << offset) + z_order_encode(0, local_max)]),
            (0, b) if b == local_max => return  Ok([(4 << offset) + z_order_encode(1, local_max), (3 << offset) + z_order_encode(local_max - x - 1, local_max), (6 << offset), (4 << offset) + z_order_encode(0, local_max - 1)]),
            (a, 0) if a == local_max => return  Ok([(1 << offset) + z_order_encode(0, local_max), (4 << offset) + z_order_encode(local_max, 1), (4 << offset) + z_order_encode(local_max - 1, 0), (2 << offset) + z_order_encode(local_max, local_max)]),
            (a, b) if a == local_max && b == local_max => return  Ok([(6 << offset) + 2, (1 << offset) + z_order_encode(local_max, local_max), (4 << offset) + z_order_encode(local_max - 1, local_max), (4 << offset) + z_order_encode(local_max, local_max - 1)]),
            (0, _) => return  Ok([(4 << offset) + z_order_encode(1, y), (4 << offset) + z_order_encode(0, y + 1), z_order_encode(local_max - y - 1,local_max), (4 << offset) + z_order_encode(0, y - 1)]),
            (a, _) if a == local_max => return  Ok([(1 << offset) + z_order_encode(y, local_max), (4 << offset) + z_order_encode(local_max, y + 1), (4 << offset) + z_order_encode(local_max - 1, y), (4 << offset) + z_order_encode(local_max, y - 1)]),
            (_, 0) => return  Ok([(4 << offset) + z_order_encode(x + 1, 0), (4 << offset) + z_order_encode(x, 1), (4 << offset) + z_order_encode(x - 1, 0), (2 << offset) + z_order_encode(x, local_max)]),
            (_, b) if b == local_max => return  Ok([(4 << offset) + z_order_encode(x + 1, local_max), (3 << offset) + z_order_encode(local_max - x - 1, local_max), (4 << offset) + z_order_encode(x - 1, local_max), (4 << offset) + z_order_encode(x, local_max - 1)]),
            _ => {}
        },
        5 => match (x, y) {
            (0, 0) => return  Ok([(5 << offset) + z_order_encode(1, 0), (5 << offset) + z_order_encode(0, 1), 0, (3 << offset) + z_order_encode(local_max, 0)]),
            (0, b) if b == local_max => return  Ok([(5 << offset) + z_order_encode(1, local_max), z_order_encode(local_max, 0), z_order_encode(local_max, 0), (5 << offset) + z_order_encode(0, local_max - 1)]),
            (a, 0) if a == local_max => return  Ok([(1 << offset) + z_order_encode(local_max - 1, 0), (5 << offset) + z_order_encode(local_max, 1), (5 << offset) + z_order_encode(local_max - 1, 0), (3 << offset)]),
            (a, b) if a == local_max && b == local_max => return  Ok([(6 << offset) + 1, (2 << offset) + z_order_encode(local_max - 1, 0), (5 << offset) + z_order_encode(local_max - 1, local_max), (5 << offset) + z_order_encode(local_max, local_max - 1)]),
            (0, _) => return  Ok([(5 << offset) + z_order_encode(1, y), (5 << offset) + z_order_encode(0, y + 1), z_order_encode(y, 0), (5 << offset) + z_order_encode(0, y - 1)]),
            (a, _) if a == local_max => return  Ok([(1 << offset) + z_order_encode(local_max - y - 1, 0), (5 << offset) + z_order_encode(local_max, y + 1), (5 << offset) + z_order_encode(local_max - 1, y), (5 << offset) + z_order_encode(local_max, y - 1)]),
            (_, 0) => return  Ok([(5 << offset) + z_order_encode(x + 1, 0), (5 << offset) + z_order_encode(x, 1), (5 << offset) + z_order_encode(x - 1, 0), (3 << offset) + z_order_encode(local_max - x, 0)]),
            (_, b) if b == local_max => return  Ok([(5 << offset) + z_order_encode(x + 1, local_max), (2 << offset) + z_order_encode(x - 1, 0), (5 << offset) + z_order_encode(x - 1, local_max), (5 << offset) + z_order_encode(x, local_max - 1)]),
            _ => {}
        },
        6 => match in_face_index {
            0 => return  Ok([(6 << offset) + 2, z_order_encode(0, local_max), (3 << offset) + z_order_encode(local_max, local_max), (4 << offset) + z_order_encode(0, local_max)]),
            1 => return  Ok([(1 << offset), (2 << offset) + z_order_encode(local_max, 0), (5 << offset) + z_order_encode(local_max, local_max), (1 << offset)]),
            _ => {}
        },
        _ => {}
    }

    // Non-exception case
    face_id = face_id << offset;
    return Ok([
        face_id + z_order_encode(x+1, y),
        face_id + z_order_encode(x, y+1), 
        face_id + z_order_encode(x-1, y), 
        face_id + z_order_encode(x, y-1)
    ]);
}



