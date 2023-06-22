import numpy as np

# This class of model is characterized by the following parameters -
# ber - bit error rate = count of faulty bits / total bits at a given voltage
# prob - likelihood of a faulty bit cell being faulty -- i.e likelihood of a
# bit cell being faulty on repeated access -- is it a transient fault ?
# ber0 - fraction of faulty bit cells that default to 0. (ber1 = ber - ber0)
# Assume each bit is likely to be faulty; ie sample from a uniform
# distribution to generate a spatial distribution of faults


class RandomFaultModel:
    MEM_ROWS = 8192
    MEM_COLS = 128
    prob = 1.0  # temporal likelihood of a given bit failing for a given access
    ber0 = 0.5
    voltage = 0
    #   BitErrorRate = [0.01212883, 0.00397706, 0.001214473, 0.00015521,
    #   0.000126225, 4.06934E-05, 1.3119E-05] # Count of faulty
    #   bits / total bits for 7 operating points.

    def __init__(self, ber, prec, pos, seed):
        self.ber = ber
        self.ber0 = RandomFaultModel.ber0
        self.precision = prec
        self.MEM_ROWS = RandomFaultModel.MEM_ROWS
        self.MEM_COLS = RandomFaultModel.MEM_COLS
        #print(
        #    "Bit Error Rate %.3f Precision %d Position %d"
        #    % (self.ber, self.precision, pos)
        #)
        if pos == -1:
            # self.BitErrorMap_flip0, self.BitErrorMap_flip1 =
            # self.ReadBitErrorMap()
            (
                self.BitErrorMap_flip0,
                self.BitErrorMap_flip1,
            ) = self.GenBitErrorMap(seed)
        else:
            (
                self.BitErrorMap_flip0,
                self.BitErrorMap_flip1,
            ) = self.GenBitPositionErrorMap(pos)

    def GenBitErrorMap(self, seed):
        bitmap = np.zeros((self.MEM_ROWS, self.MEM_COLS))
        bitmap_flip0 = np.zeros((self.MEM_ROWS, self.MEM_COLS))
        bitmap_flip1 = np.zeros((self.MEM_ROWS, self.MEM_COLS))

        if seed is not None:
            np.random.seed(seed)
        bitmap_t = np.random.rand(self.MEM_ROWS, self.MEM_COLS)
        bitmap[bitmap_t < self.ber] = 1

        # print(bitmap)
        if seed is not None:
            np.random.seed(seed + 1)
        bitmap_flip = np.random.rand(self.MEM_ROWS, self.MEM_COLS)

        bitmap_flip0[bitmap_flip < self.ber0] = 1
        bitmap_flip1[bitmap_flip >= self.ber0] = 1
        # print(bitmap_flip0)
        # print(bitmap_flip1)
        bitmap_flip0 = bitmap * bitmap_flip0
        bitmap_flip1 = bitmap * bitmap_flip1

        bitmap_flip0 = bitmap_flip0.astype(np.int64)
        bitmap_flip1 = bitmap_flip1.astype(np.int64)
        # print(bitmap_flip0)
        # print(bitmap_flip1)
        bitcells = self.MEM_ROWS * self.MEM_COLS
        # print("Read 0 Bit Error Rate", sum(sum(bitmap_flip0)) / bitcells)
        # print("Read 1 Bit Error Rate", sum(sum(bitmap_flip1)) / bitcells)
        return bitmap_flip0, bitmap_flip1

    def GenBitPositionErrorMap(self, pos):

        bitmap = np.zeros((self.MEM_ROWS, self.MEM_COLS))
        bitmap_flip0 = np.zeros((self.MEM_ROWS, self.MEM_COLS))
        bitmap_flip1 = np.zeros((self.MEM_ROWS, self.MEM_COLS))

        # Generate errors at rate ber in a specific bit position,
        # maximum of one error per weight in the specified position
        weights_per_row = int(self.MEM_COLS / self.precision)
        bitmap_pos = np.zeros((self.MEM_ROWS, weights_per_row))
        bitmap_t = np.random.rand(self.MEM_ROWS, weights_per_row)
        bitmap_pos[bitmap_t < self.ber] = 1
        # Insert the faulty column in bit error map
        for k in range(0, weights_per_row):
            bitmap[:, k * self.precision + pos] = bitmap_pos[:, k]
        # print(bitmap)

        bitmap_flip = np.random.rand(self.MEM_ROWS, self.MEM_COLS)
        bitmap_flip0[bitmap_flip < self.ber0] = 1
        bitmap_flip1[bitmap_flip >= self.ber0] = 1
        # print(bitmap_flip0)
        # print(bitmap_flip1)
        bitmap_flip0 = bitmap * bitmap_flip0
        bitmap_flip1 = bitmap * bitmap_flip1

        # print(bitmap_flip0)
        # print(bitmap_flip1)
        bitmap_flip0 = bitmap_flip0.astype(np.uint32)
        bitmap_flip1 = bitmap_flip1.astype(np.uint32)
        bitcells = self.MEM_ROWS * self.MEM_COLS
        print(
            "Bit Error Rate",
            sum(sum(bitmap_flip0)) / bitcells
            + sum(sum(bitmap_flip1)) / bitcells,
        )
        return bitmap_flip0, bitmap_flip1

    def ReadBitErrorMap(self):
        mem_voltage = self.voltage
        chip = "n"
        fname = (
            "./faultmaps_chip_"
            + chip
            + "/fmap_sa0_v_"
            + str(mem_voltage)
            + ".txt"
        )
        some_arr = np.genfromtxt(fname, dtype="uint32", delimiter=",")
        bitmap_flip0 = some_arr[0 : self.MEM_ROWS, 0 : self.MEM_COLS]
        print(
            "SA 0 Bit error rate",
            (bitmap_flip0.sum() / (self.MEM_ROWS * self.MEM_COLS)),
        )
        fname = (
            "./faultmaps_chip_"
            + chip
            + "/fmap_sa1_v_"
            + str(mem_voltage)
            + ".txt"
        )
        some_arr = np.genfromtxt(fname, dtype="uint32", delimiter=",")
        bitmap_flip1 = some_arr[0 : self.MEM_ROWS, 0 : self.MEM_COLS]
        print(
            "SA 1 Bit error rate",
            (bitmap_flip1.sum() / (self.MEM_ROWS * self.MEM_COLS)),
        )
        return bitmap_flip0, bitmap_flip1
