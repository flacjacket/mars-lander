require 'torch'
torch.setdefaulttensortype('torch.DoubleTensor') -- This is the default.  It is here as a reminder.  Switch to FloatTensor for CNN.
require 'nn'
require 'image'
gm = require 'graphicsmagick'
require 'xlua'  -- progress bar
require 'optim'  -- confusion matrix; gradient decent optimization
require 'paths'  -- read OS directory structure


-- read in data

--[[
file_stream = io.open('../train/terrainS4C4R20_100/terrainS4C4R20_100_500by500.ply')
ply = file_stream:read("*a")
file_stream:close()


local _, begin_match = string.find(ply, 'element vertex')
local num_vertices = string.match(ply, '%d+', begin_match)

local resolution = math.sqrt(num_vertices)
local topograph = torch.DoubleTensor(1, resolution, resolution)  -- floatTensor is not precise enough until after normalization

_, begin_match = string.find(ply, 'end_header')
for i = 1,num_vertices do
	local _, next_match = string.find(ply, '%-*%d+%.*%d*%s%-*%d+%.*%d*%s', begin_match + 1)
	local z_coord = string.match(ply, '%-*%d+%.*%d*', next_match + 1)
	topograph[1][math.ceil(i/resolution)][1+(i-1)%resolution] = tonumber(z_coord)
	_, begin_match = string.find(ply, '%d+%s%d+%s%d+', next_match + #z_coord + 1)  -- Try reading in RGB and plotting on grid to see if it visually matches PGM
end
--]]

print '==> loading image'

-- read in data
local topograph = gm.load('../train/terrainS4C4R20_100/terrainS4C4R20_100_dem.bmp', 'double')

print '==> loading solution map'

-- read in solution
local solution_map = gm.load('../train/terrainS4C4R20_100/terrainS4C4R20_100.invHazard.pgm', 'double')
solution_map = (nn.SpatialMaxPooling(2,2,2,2):forward(solution_map)[1]):type('torch.FloatTensor')  -- RGB channels are redundant; uses 0's and 1's



-- create the reduced data representations for sliding disks

-- DIMENSIONS in meters
local map_dim = 100
local lander_diameter = 3.4
local footpad_diameter = .5
local belly_height = .39  -- may not need this

-- PARAMETERS
local resolution = 500
local disk_radius = math.floor(resolution * 0.5 * lander_diameter / map_dim)  -- equals floor of 8.5
local disk_upscale = 2  -- make the disk bigger so that we don't sample any pixel more than once (giving it extra weight); maybe nonsense
local azimuth_dim = 32  -- 16 spokes around the half-disk
local radial_dim = 17  -- approximately one pixel per decimeter




print '==> preprocessing sub-images'

local buffered_full_disks = {}
local buffered_half_disks = {}
local cropped_solution_map = {}
local preprocessed_counter = 0
for col = disk_radius + 1, resolution - disk_radius do  -- don't consider disks which extend past borders
	for row = disk_radius + 1, resolution - disk_radius do

		preprocessed_counter = preprocessed_counter + 1
		xlua.progress(preprocessed_counter , (resolution - 2*disk_radius) * (resolution - 2*disk_radius))

		local disk = image.polar(torch.DoubleTensor(1,azimuth_dim,radial_dim),
		                         topograph:sub(1,1,col-disk_radius,col+disk_radius,row-disk_radius,row+disk_radius),
								 'bilinear', 'valid'):narrow(3,2,radial_dim-1)
		-- removed origin (Neural nets never see the origin!!!!... although there is some interpolation around origin.  Maybe add this info back in somehow?)
		-- remove origin (r=0); maybe should be after normalization?
		
		--[[
		-- buffer disk for CNN using periodicity
		local buffered_full_disk = torch.DoubleTensor(1, azimuth_dim/2 + 4, radial_dim - 1)
		buffered_full_disk:narrow(2,3,azimuth_dim/2)[1] = ( nn.SpatialMaxPooling(1,2,1,2):forward(disk) )[1]:clone()
		buffered_full_disk:narrow(2,1,2)[1] = buffered_full_disk:narrow(2,azimuth_dim/2+1,2)[1]:clone()
		buffered_full_disk:narrow(2,azimuth_dim/2+3,2)[1] = buffered_full_disk:narrow(2,3,2)[1]:clone()

		local unbuffered_full_disk_mean = torch.mean(buffered_full_disk:narrow(2,3,azimuth_dim/2))
		local unbuffered_full_disk_std = torch.std(buffered_full_disk:narrow(2,3,azimuth_dim/2))
		--]]

		-- create buffered half disks (using relative altitudes of polar opposite positions of lander)
		local buffered_half_disk = torch.DoubleTensor(1, azimuth_dim/2 + 4, radial_dim - 1)	
		local disk_chunks = disk:chunk(2,2)


		buffered_half_disk:narrow(2,3,azimuth_dim/2)[1] = (disk_chunks[1] - disk_chunks[2])[1]:clone()  -- maybe eliminate next lines by chunking buffered disk?
		buffered_half_disk:narrow(2,1,2)[1] = - buffered_half_disk:narrow(2,azimuth_dim/2+1,2)[1]:clone()  -- minus because relationship M[theta]-M[theta+pi] is flipped
		buffered_half_disk:narrow(2,azimuth_dim/2+3,2)[1] = - buffered_half_disk:narrow(2,3,2)[1]:clone()

		local unbuffered_half_disk_mean = torch.mean(buffered_half_disk:narrow(2,3,azimuth_dim/2))
		local unbuffered_half_disk_std = torch.std(buffered_half_disk:narrow(2,3,azimuth_dim/2))

		--table.insert(preprocessed_images, ( (buffered_half_disk - unbuffered_mean) / unbuffered_std ):type('torch.FloatTensor'))
		--table.insert(buffered_full_disks, ( (buffered_full_disk - unbuffered_full_disk_mean) / unbuffered_full_disk_std ):type('torch.FloatTensor'))
		table.insert(buffered_half_disks, ( (buffered_half_disk - unbuffered_half_disk_mean) / unbuffered_half_disk_std ):type('torch.FloatTensor'))
		table.insert(cropped_solution_map, 1 + solution_map[col][row])	-- 0 -> 1, 1 -> 2  (index labels for output)
			
	end
end


-- save preprocessed data

print '==> saving preprocessed data to disk'

--torch.save('preprocessed.dat', preprocessed_images)
torch.save('half_disks.dat', buffered_half_disks)
--torch.save('full_disks.dat', buffered_full_disks)
torch.save('solution.dat', cropped_solution_map)

