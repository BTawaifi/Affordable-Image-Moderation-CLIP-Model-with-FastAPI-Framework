import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { AxiosResponse } from 'axios';
import * as FormData from 'form-data';
import { MulterFile } from 'multer';

@Injectable()
export class ClipService {
  private readonly logger = new Logger(ClipService.name);
  private readonly apiUrl = 'http://127.0.0.1:8500';

  constructor(private httpService: HttpService) {}

  async classifyImage(
    image: MulterFile,
    classes: string,
  ): Promise<{
    clipPrediction: string;
    clipLabelProbabilities: { [key: string]: number };
  }> {
    try {
      const formData = new FormData();

      formData.append('image', image.buffer, {
        filename: image.originalname,
        contentType: image.mimetype,
      });
      classes && formData.append('classes', classes);

      const response: AxiosResponse = await this.httpService
        .post(`${this.apiUrl}/classify_image`, formData, {
          headers: formData.getHeaders(),
        })
        .toPromise();

      return response.data;
    } catch (error) {
      this.logger.error(`Error classifying image: ${(error as Error).message}`);
      throw new Error('Failed to classify image');
    }
  }
}
