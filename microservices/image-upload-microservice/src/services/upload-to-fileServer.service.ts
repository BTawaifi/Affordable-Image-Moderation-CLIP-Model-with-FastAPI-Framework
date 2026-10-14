import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { AxiosResponse } from 'axios';
import * as FormData from 'form-data';
import { ImageUploadException } from '../filters/exceptions/image-upload.exception';

@Injectable()
export class UploadToFileServerService {
  private readonly logger = new Logger(UploadToFileServerService.name);
  private readonly apiUrl = 'http://127.0.0.1:8500';

  constructor(private httpService: HttpService) {}

  async upload(image: any, originalName): Promise<{
    message: string;
    filename: string;
    size: string;
    location: string;
  }> {
    try {
      const formData = new FormData();
      
      formData.append('image', image, {
        filename: 'temp.webp',
        contentType: 'image/webp',
      });
      
      const response: AxiosResponse = await this.httpService
        .post(`http://localhost:5030/file-server/upload`, formData, {
          headers: formData.getHeaders(),
        })
        .toPromise();

      return response.data;
    } catch (error) {
      this.logger.error(`Error uploading image: ${(error as Error).message}`);
      throw new ImageUploadException('Failed to upload image');
    }
  }
}
