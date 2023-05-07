import { diskStorage } from 'multer';
import { extname } from 'path';

export const multerConfig = {
  storage: diskStorage({
    destination: './uploads',
    filename: (req, file, callback) => {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
      const fileExtension = extname(file.originalname);
      callback(null, `${uniqueSuffix}${fileExtension}`);
    },
  }), 
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB (adjust the value as needed)
  },
};
