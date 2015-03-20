require 'torch'
torch.setdefaulttensortype('torch.DoubleTensor') -- This is the default.  It is here as a reminder.  Switch to FloatTensor for CNN.
require 'nn'
require 'image'
require 'xlua'  -- progress bar
require 'optim'  -- confusion matrix; gradient decent optimization
require 'paths'  -- read OS directory structure


-- read in data

file_stream = io.open('../train/terrainS4C4R20_100/terrainS4C4R20_100_500by500.ply')
ply = file_stream:read("*a")
file_stream:close()


local _, begin_match = string.find(ply, 'element vertex')
local num_vertices = string.match(ply, '%d+', begin_match)

local resolution = math.sqrt(num_vertices)
topograph = torch.DoubleTensor(1, resolution, resolution)  -- floatTensor is not precise enough until after normalization

_, begin_match = string.find(ply, 'end_header')
for i = 1,num_vertices do
	local _, next_match = string.find(ply, '%-*%d+%.*%d*%s%-*%d+%.*%d*%s', begin_match + 1)
	local z_coord = string.match(ply, '%-*%d+%.*%d*', next_match + 1)
	topograph[1][math.ceil(i/resolution)][1+(i-1)%resolution] = tonumber(z_coord)
	_, begin_match = string.find(ply, '%d+%s%d+%s%d+', next_match + #z_coord + 1)
end




-- create the reduced data representations for sliding disks

-- DIMENSIONS in meters
local map_dim = 100
local lander_diameter = 3.4
local footpad_diameter = .5
local belly_height = .39  -- may not need this

-- PARAMETERS
local disk_radius = math.floor(resolution * 0.5 * lander_diameter / map_dim)  -- equals floor of 8.5
local disk_upscale = 2  -- make the disk bigger so that we don't sample any pixel more than once (giving it extra weight); maybe nonsense
local azimuth_dim = 32  -- 16 spokes around the half-disk
local radial_dim = 17  -- approximately one pixel per decimeters


preprocessed_images = {}
for col = disk_radius + 1, resolution - disk_radius do  -- don't consider disks which extend past borders
	for row = disk_radius + 1, resolution - disk_radius do

		local disk = image.polar(torch.Tensor(1,azimuth_dim,radial_dim),
		                         topograph:sub(1,1,col-disk_radius,col+disk_radius,row-disk_radius,row+disk_radius),
								 'bilinear', 'valid')
		local disk_chunks = disk:chunk(2,2)
		
		local buffered_half_disk = torch.Tensor(1, azimuth_dim/2 + 4, radial_dim - 1)
		
		-- remove origin (r=0); maybe should be after normalization?
		buffered_half_disk:narrow(2,3,azimuth_dim/2)[1] = ( (disk_chunks[1] - disk_chunks[2]):narrow(3,2,radial_dim-1) )[1]:clone()
		buffered_half_disk:narrow(2,1,2)[1] = - buffered_half_disk:narrow(2,azimuth_dim/2+1,2)[1]:clone()
		buffered_half_disk:narrow(2,azimuth_dim/2+3,2)[1] = - buffered_half_disk:narrow(2,3,2)[1]:clone()

		local unbuffered_mean = torch.mean(buffered_half_disk:narrow(2,3,azimuth_dim/2))
		local unbuffered_std = torch.std(buffered_half_disk:narrow(2,3,azimuth_dim/2))

		table.insert(preprocessed_images, ( (buffered_half_disk - unbuffered_mean) / unbuffered_std ):type('torch.FloatTensor'))		
	end
end


-- save preprocessed data

torch.save('preprocessed.dat', preprocessed_images)


