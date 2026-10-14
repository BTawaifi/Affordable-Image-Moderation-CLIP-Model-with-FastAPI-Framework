import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { AxiosResponse } from 'axios';
import * as FormData from 'form-data';
import { MulterFile } from 'multer';

@Injectable()
export class ResnetService {
  private readonly logger = new Logger(ResnetService.name);
  private readonly apiUrl = 'http://127.0.0.1:8200';

  constructor(private httpService: HttpService) {}

  async classifyImage(image: MulterFile): Promise<{
    resnetPrediction: string;
    resnetLabelProbabilities: string;
  }> {
    try {
      const formData = new FormData();

      formData.append('image', image.buffer, {
        filename: image.originalname,
        contentType: image.mimetype,
      });

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
